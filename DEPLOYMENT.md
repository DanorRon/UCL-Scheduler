# Railway Deployment Guide

This guide will help you deploy your UCL Scheduler to Railway.

## Prerequisites

1. **GitHub Account** - Your code needs to be on GitHub
2. **Railway Account** - Sign up at [railway.app](https://railway.app)
3. **Google Sheets API Credentials** - Your service account JSON file

## Step 1: Prepare Your Code

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

2. **Convert Credentials** (run this locally):
   ```bash
   python convert_credentials.py
   ```
   This will give you a JSON string to use as an environment variable.

## Step 2: Deploy to Railway

1. **Go to [railway.app](https://railway.app)**
2. **Sign in with GitHub**
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your repository**
6. **Railway will automatically detect it's a Flask app and deploy**

## Step 3: Set Environment Variables

After deployment, add your Google Sheets credentials:

1. **Go to your Railway project dashboard**
2. **Click "Variables" tab**
3. **Add new variable**:
   - **Name**: `GOOGLE_SHEETS_CREDENTIALS`
   - **Value**: (paste the JSON string from `convert_credentials.py`)
4. **Click "Add"**
5. **Railway will automatically redeploy**

## Step 4: Get Your Public URL

Railway will give you a URL like:
```
https://your-app-name.railway.app
```

Your scheduler is now live and publicly accessible!

## Troubleshooting

### Common Issues:

1. **"Credentials error"** - Make sure the environment variable is set correctly
2. **"Invalid JSON"** - Check that the JSON string is valid (no line breaks)
3. **"Failed to retrieve data"** - Verify your Google Sheets URL and permissions

### Debugging:

1. **Check Railway logs** in the dashboard
2. **Test locally** with environment variables:
   ```bash
   export GOOGLE_SHEETS_CREDENTIALS='{"your":"json","here":"..."}'
   python -m ucl_scheduler.web_interface.app
   ```

## Security Notes

- ✅ Credentials are stored securely in Railway
- ✅ No sensitive files in your Git repository
- ✅ Environment variables are encrypted
- ✅ HTTPS is automatic

## Cost

- **Free tier**: $5 credit monthly (plenty for personal use)
- **No credit card required** to start
- **Automatic renewals** of free credit 