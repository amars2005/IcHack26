# AI Custom Situation Setup

The TacticsBoard app can generate custom tactical situations using AI. This feature uses OpenAI's GPT-4o-mini model for reliable and cost-effective generation.

## Setup Instructions

### 1. Get an OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy your API key (starts with `sk-`)

### 2. Configure the API Key

Open your browser console (F12) and run:

```javascript
localStorage.setItem('openai_api_key', 'sk-YOUR-API-KEY-HERE');
```

Replace `sk-YOUR-API-KEY-HERE` with your actual API key.

### 3. Use the Feature

1. Open the menu (arrow button in top-right)
2. Scroll to "Custom Situation" section
3. Describe an attacking scenario, for example:
   - "Fast break with 3 attackers vs 2 defenders"
   - "Corner kick with all players packed in the box"
   - "Counter-attack after winning possession in midfield"
   - "Throw-in near the penalty area with numerical advantage"
4. Click "Generate with AI"
5. Wait for the positions to be generated (typically 2-5 seconds)

## How It Works

The system uses carefully engineered prompts that:
- Provide exact pitch specifications (120m x 80m)
- Define the coordinate system clearly
- Specify tactical guidelines for realistic positioning
- Request structured JSON output for reliability
- Validate and clamp all positions to pitch boundaries

The LLM understands:
- Standard formations and player roles
- Tactical positioning for various situations
- Realistic spacing and movement patterns
- Set piece configurations
- Phase of play (attack, defense, transition)

## Cost

GPT-4o-mini is very affordable:
- Input: $0.150 per 1M tokens
- Output: $0.600 per 1M tokens
- Typical request: ~$0.0005-0.001 per generation

With the free $5 credit for new accounts, you can generate thousands of situations.

## Troubleshooting

**"Generation failed. Check API key."**
- Verify your API key is correctly set in localStorage
- Check you have credits remaining in your OpenAI account
- Ensure your API key has permission to use the chat completions endpoint

**Positions seem unrealistic:**
- Try being more specific in your description
- Mention the phase of play (attacking, defending, transition)
- Specify key tactical elements (numbers up, overload, etc.)

**Slow generation:**
- This is normal; AI generation takes 2-5 seconds
- Don't click the button multiple times

## Privacy & Security

- API key is stored in browser localStorage (client-side only)
- Never commit your API key to version control
- API calls go directly from browser to OpenAI (not through our servers)
- Consider using API key rotation for production use
