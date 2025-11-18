# How to Set Environment Variables in Render

## Step-by-Step Guide

### 1. Go to Render Dashboard
- Visit: https://dashboard.render.com
- Log in to your account

### 2. Select Your Service
- Click on your **IbtikarAI** service
- (The service that's deployed at `ibtikarai.onrender.com`)

### 3. Navigate to Environment Tab
- In the service page, look for tabs at the top:
  - **Overview** | **Logs** | **Metrics** | **Environment** | **Settings**
- Click on **Environment**

### 4. Add Environment Variable
- Scroll down to the **Environment Variables** section
- Click the **"Add Environment Variable"** button (or **"Add"** button)

### 5. Enter the Values
- **Key**: `HUGGINGFACE_MODEL_ID`
- **Value**: `distilbert-base-multilingual-cased`
- Click **Save Changes**

### 6. Wait for Redeploy
- Render will automatically detect the change
- It will show "New deploy available" or start redeploying automatically
- Wait 2-3 minutes for the redeploy to complete

## Visual Guide

```
Render Dashboard
└── Your Services
    └── IbtikarAI (click this)
        └── Environment Tab (click this)
            └── Environment Variables Section
                └── Add Environment Variable (click this)
                    ├── Key: HUGGINGFACE_MODEL_ID
                    └── Value: distilbert-base-multilingual-cased
                        └── Save Changes
```

## Alternative: Using Render CLI

If you prefer command line:

```bash
# Install Render CLI
npm install -g render-cli

# Set environment variable
render env:set HUGGINGFACE_MODEL_ID=distilbert-base-multilingual-cased --service ibtikarai
```

## Verify It's Set

After setting, you can verify:
1. Go back to **Environment** tab
2. You should see:
   ```
   HUGGINGFACE_MODEL_ID = distilbert-base-multilingual-cased
   ```

## What Happens Next

1. Render detects the environment variable change
2. Automatically triggers a new deployment
3. Your code will use `distilbert-base-multilingual-cased` instead of the default
4. This model is ~250MB (smallest option) and should work on 512MB free tier

## Other Model Options

You can also use:
- `textdetox/bert-multilingual-toxicity-classifier` (default, ~400MB, pre-trained for toxicity)
- `aubmindlab/bert-base-arabertv2` (largest, ~500MB, may cause memory issues)

Just change the **Value** field to any of these model IDs.

