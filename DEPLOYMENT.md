# Deployment Instructions for Streamlit Cloud

## Problem
Streamlit Cloud is using Python 3.13.9 by default, but MediaPipe doesn't support Python 3.13 yet.

## Solution

### Option 1: Manual Configuration (Recommended)
1. Go to https://share.streamlit.io/
2. Find your app "sign-language-classification"
3. Click the menu (⋮) → Settings
4. Look for "Python version" or "Advanced settings"
5. Set Python version to **3.11**
6. Save and reboot the app

### Option 2: Delete and Redeploy
1. Delete the current deployment
2. Redeploy from GitHub
3. During setup, select Python 3.11 if prompted

### Option 3: Contact Streamlit Support
If the above options don't work, the issue might be:
- Streamlit Cloud's default Python version changed to 3.13
- `runtime.txt` file is not being respected
- Platform bug

Contact Streamlit support at: https://discuss.streamlit.io/

## Files We've Created
- `runtime.txt` - Specifies Python 3.11
- `.python-version` - Alternative Python version specification
- `pyproject.toml` - Python version requirement
- `packages.txt` - System dependencies for OpenCV
- `requirements.txt` - Pinned package versions

## Expected Behavior
Once Python 3.11 is used, all packages should install successfully:
- streamlit==1.28.0
- opencv-python-headless==4.8.1.78
- mediapipe==0.10.9
- tensorflow==2.15.0
- And other dependencies

## Verification
After deployment, check the logs for:
```
Using Python 3.11.x environment
```

Instead of:
```
Using Python 3.13.9 environment
```
