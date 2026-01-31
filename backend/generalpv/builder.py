import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from loader import StatsBombLoader
from parser import MatchParser


def _parse_single_match(m_id):
    """Parse a single match - module level function for multiprocessing."""
    import traceback
    try:
        parser = MatchParser(m_id)
        return m_id, parser.parse(), None
    except Exception as e:
        return m_id, None, f"{e}\n{traceback.format_exc()}"


class DatasetBuilder:
    def __init__(self, max_workers=8):
        self.loader = StatsBombLoader()
        self.max_workers = max_workers
        
    def build_dataset(self, limit=None):
        match_ids = self.loader.get_all_360_matches()
        
        if limit:
            match_ids = match_ids[:limit]
            print(f"Limiting to first {limit} matches for testing.")
        
        total_matches = len(match_ids)
        print(f"Processing {total_matches} matches with {self.max_workers} workers...")
        
        master_frames = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_parse_single_match, m_id) for m_id in match_ids]
            
            for future in as_completed(futures):
                completed += 1
                
                if completed % 10 == 0 or completed == total_matches:
                    print(f"Completed {completed}/{total_matches} matches...")
                
                m_id, df_match, error = future.result()
                if error:
                    print(f"Error parsing match {m_id}: {error}")
                elif df_match is not None:
                    master_frames.append(df_match)
                
        if not master_frames:
            return pd.DataFrame()
        
        # Concat with ignore_index=True - each df already has clean indices
        full_df = pd.concat(master_frames, ignore_index=True)
        
        # Filter to only keep matches that have at least one event with player position data
        # Check if p0_x is non-null as indicator of having 360 data
        matches_with_360 = full_df[full_df['p0_x'].notna()]['match_id'].unique()
        matches_before = full_df['match_id'].nunique()
        full_df = full_df[full_df['match_id'].isin(matches_with_360)]
        matches_after = full_df['match_id'].nunique()
        
        print(f"Filtered matches: {matches_before} -> {matches_after} (kept only matches with 360 data)")
        
        # --- FINAL SORT ---
        full_df['timestamp_dt'] = pd.to_timedelta(full_df['timestamp'], errors='coerce')
        full_df = full_df.sort_values(by=['match_id', 'period', 'timestamp_dt', 'index'])
        full_df = full_df.drop(columns=['timestamp_dt'])
        
        return full_df

    def process_chains(self, df):
        """
        Takes the master dataframe and adds Chain IDs and Goal outcomes.
        """
        print("Processing possession chains...")
        
        # Ensure strict sorting
        df = df.sort_values(by=['match_id', 'period', 'index']).reset_index(drop=True)
        
        # 1. Detect Breaks
        match_change = df['match_id'] != df['match_id'].shift(1)
        period_change = df['period'] != df['period'].shift(1)
        team_change = df['team_id'] != df['team_id'].shift(1)
        prev_was_shot = df['type'].shift(1) == 'Shot'
        
        is_new_chain = match_change | period_change | team_change | prev_was_shot
        
        # 2. Assign Chain IDs
        df['chain_id'] = is_new_chain.cumsum()
        
        # 3. Determine Goal Outcomes
        is_goal = (df['type'] == 'Shot') & (df['success'] == 1)
        goal_chains = df.loc[is_goal, 'chain_id'].unique()
        
        df['chain_goal'] = 0
        df.loc[df['chain_id'].isin(goal_chains), 'chain_goal'] = 1
        
        print(f"Processed {df['chain_id'].max()} unique chains.")
        return df