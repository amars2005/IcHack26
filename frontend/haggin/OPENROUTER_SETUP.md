# OpenRouter API Setup (FREE)

The TacticsBoard app uses OpenRouter to access free AI models for generating custom tactical situations.

## Why OpenRouter?

- **100% FREE** models available (Google Gemini Flash, Meta Llama, etc.)
- No credit card required for free models
- Access to multiple AI providers through one API
- More generous rate limits than direct provider APIs

## Quick Setup (2 minutes)

### 1. Get a FREE OpenRouter API Key

1. Go to https://openrouter.ai/
2. Click "Sign In" (top right)
3. Sign in with Google, GitHub, or email
4. Go to https://openrouter.ai/keys
5. Click "Create Key"
6. Give it a name (e.g., "TacticsBoard")
7. Copy your API key (starts with `sk-or-v1-`)

**No credit card required!** Free models are available immediately.

### 2. Configure the API Key

Open your browser at http://localhost:5175/ and press **F12** to open the console, then run:

```javascript
localStorage.setItem('openrouter_api_key', 'sk-or-v1-YOUR-KEY-HERE');
```

Replace `sk-or-v1-YOUR-KEY-HERE` with your actual API key.

### 3. Refresh and Use

1. Refresh the page (F5)
2. Open the menu (← arrow in top-right)
3. Scroll to "Custom Situation"
4. Try: "Fast counter-attack with 4 attackers vs 3 defenders"
5. Click "Generate with AI"

## Example Prompts

```
"Corner kick with a short corner routine"
"Overload on the left wing with overlapping fullback"
"Quick throw-in near the penalty box"
"3v2 situation after winning the ball in midfield"
"Free kick 25 yards out with a 5-man wall"
"Counter-attack with the defense pushed high up"
"Build-up play from the back with goalkeeper involved"
```

## Current Model

We use **Google Gemini Flash 1.5 (8B)** which is:
- ✅ Completely FREE
- ✅ Fast (2-3 second responses)
- ✅ High quality tactical understanding
- ✅ Good at following structured output formats

## Alternative Free Models

You can modify `src/llm.ts` to use other free models:

```typescript
model: 'meta-llama/llama-3.1-8b-instruct:free'  // Meta's Llama
model: 'mistralai/mistral-7b-instruct:free'     // Mistral
model: 'google/gemini-flash-1.5-8b'             // Google (current)
```

See all free models at: https://openrouter.ai/models?max_price=0

## Rate Limits (Free Tier)

- ~200 requests per day per IP
- ~10 requests per minute
- More than enough for tactical training!

## Troubleshooting

**"Generation failed. Check API key."**
- Make sure you set `openrouter_api_key` (not `openai_api_key`)
- Verify the key starts with `sk-or-v1-`
- Check console (F12) for detailed error messages

**"Rate limit exceeded"**
- Free models have rate limits
- Wait a minute and try again
- Consider upgrading to paid tier for unlimited access

**Positions seem off:**
- Be more specific in your description
- Mention: phase of play, numerical advantage, specific positions
- Example: "3 attackers (2 wingers + striker) vs 2 center backs"

## Privacy & Security

- ✅ API key stored locally in your browser only
- ✅ Requests go directly from your browser to OpenRouter
- ✅ No data stored on our servers
- ✅ OpenRouter doesn't train models on your data by default

## Cost (Optional Paid Models)

While we use free models, OpenRouter also offers premium models:
- GPT-4: ~$0.03 per generation
- Claude Opus: ~$0.015 per generation
- But free models work great for this use case!

## Need Help?

Check the browser console (F12) for detailed error messages or reach out with the specific error you're seeing.
