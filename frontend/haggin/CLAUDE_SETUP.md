# Claude API Setup

The TacticsBoard app uses Claude (Anthropic) for generating custom tactical situations.

## Quick Setup

### 1. Get Your Anthropic API Key

If you already have Claude credits:
1. Go to https://console.anthropic.com/settings/keys
2. Click "Create Key"
3. Copy your API key (starts with `sk-ant-`)

### 2. Set the API Key

Open http://localhost:5175/ and press **F12** to open the console, then run:

```javascript
localStorage.setItem('anthropic_api_key', 'sk-ant-YOUR-KEY-HERE');
```

Replace `sk-ant-YOUR-KEY-HERE` with your actual API key.

### 3. Refresh and Use

1. Refresh the page (F5)
2. Open menu (← arrow in top-right)
3. Scroll to "Custom Situation"
4. Try: "Fast counter-attack with 4 attackers vs 3 defenders"
5. Click "Generate with AI"

## Current Model

We use **Claude 3.5 Haiku** which is:
- ⚡ Fast (1-2 second responses)
- 💰 Cost-effective ($0.001 per generation)
- 🎯 High quality tactical understanding
- ✅ Excellent at structured JSON output

## Example Prompts

```
"Corner kick with players making near-post runs"
"Overload on the right wing with inverted winger"
"Build-up from goalkeeper with high press"
"Quick throw-in creating 3v2 in the box"
"Counter-attack after defensive header"
"Free kick 30 yards out with decoy runners"
"Transition moment after winning ball in final third"
```

## Alternative Models

You can edit `src/llm.ts` line 51 to use different Claude models:

```typescript
model: 'claude-3-5-haiku-20241022'   // Fastest, cheapest (current)
model: 'claude-3-5-sonnet-20241022'  // Best quality, slightly more expensive
```

## Costs (with your Claude credits)

- **Haiku**: ~$0.001 per generation (1000 generations per $1)
- **Sonnet**: ~$0.003 per generation (300 generations per $1)

Haiku is perfect for this use case - fast and accurate!

## Why Claude?

- 🔒 **Reliable parsing**: Claude excels at structured output
- 🧠 **Tactical understanding**: Great at understanding football concepts
- ⚡ **Fast responses**: 1-2 seconds for Haiku
- 📊 **Consistent format**: Returns clean JSON every time

## Troubleshooting

**"Generation failed. Check API key."**
- Verify you set `anthropic_api_key` (not `openai_api_key` or `openrouter_api_key`)
- Check the key starts with `sk-ant-`
- Verify you have credits remaining in your Anthropic account
- Check browser console (F12) for detailed error

**Positions seem unrealistic:**
- Be more specific: mention phase of play, formation, numerical advantage
- Example: "3 forwards pressing high against 2 center backs playing out from the back"

**Response parsing error:**
- This should be rare with Claude (it's very good at structured output)
- Check console for the actual response
- File an issue if this persists

## Privacy & Security

- ✅ API key stored locally in browser only
- ✅ Requests go directly from browser to Anthropic
- ✅ No data stored on our servers
- ✅ Your generations are not used for training

## Rate Limits

Standard Anthropic API limits:
- Depends on your account tier
- Generally 50-100 requests per minute
- More than sufficient for tactical training

## Need Help?

Check the browser console (F12) for detailed error messages or reach out with the specific error you're seeing.
