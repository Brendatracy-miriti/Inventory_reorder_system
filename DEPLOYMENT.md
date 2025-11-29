# Streamlit Deployment Guide

## 📦 Inventory Reorder Management System

### Prerequisites
- GitHub account
- Streamlit Community Cloud account (free at https://streamlit.io/cloud)

### Deployment Steps

#### 1. Prepare Your Repository
```bash
# Navigate to your project root
cd C:\Users\USER\data_science\Capstone_project\Inventory_reorder_system\Inventory_reorder_system

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Prepare dashboard for Streamlit deployment"

# Push to GitHub
git push origin main
```

#### 2. Structure Required for Deployment
Your repository should have this structure:
```
Inventory_reorder_system/
├── Inventory_dashboard/
│   └── app.py                    # Main dashboard file
├── Data/
│   ├── master_inventory_policy.csv
│   └── retail_store_inventory.csv
├── requirements.txt               # Python dependencies (✅ Created)
└── .streamlit/
    └── config.toml               # Streamlit config (✅ Created)
```

#### 3. Deploy on Streamlit Cloud

**Option A: Deploy from GitHub**
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `Brendatracy-miriti/Inventory_reorder_system`
5. Set **Main file path**: `Inventory_dashboard/app.py`
6. Set **Branch**: `main`
7. Click "Deploy!"

**Option B: Deploy via Command Line**
```bash
# Install Streamlit CLI (if needed)
pip install streamlit

# Login to Streamlit
streamlit login

# Deploy
streamlit deploy
```

#### 4. Important Configuration

**App Settings in Streamlit Cloud:**
- **Python version**: 3.9+ (recommended 3.10 or 3.11)
- **Main file path**: `Inventory_dashboard/app.py`
- **Secrets**: None required (using local CSV files)

#### 5. Troubleshooting

**Problem: "Module not found" errors**
- Solution: Check `requirements.txt` includes all dependencies
- Run locally first: `pip install -r requirements.txt`

**Problem: "Data files not found"**
- Solution: Ensure `Data` folder is committed to GitHub
- Check `.gitignore` isn't excluding CSV files

**Problem: App loads slowly**
- Solution: Streamlit Cloud free tier has resource limits
- Consider reducing data size or upgrading to paid tier

#### 6. Local Testing Before Deployment
```bash
# Navigate to dashboard folder
cd Inventory_dashboard

# Run locally
streamlit run app.py

# Test on local network (accessible to other devices)
streamlit run app.py --server.address=0.0.0.0
```

#### 7. Update Deployed App
After deployment, any push to GitHub main branch will auto-redeploy:
```bash
git add .
git commit -m "Update dashboard features"
git push origin main
```

### 🚀 Your Dashboard URL
After deployment, Streamlit will provide a URL like:
`https://[your-app-name].streamlit.app`

Share this URL with stakeholders for instant access!

---

### 📊 Features Available After Deployment
✅ Real-time inventory monitoring
✅ Interactive filters and date ranges
✅ Visual reports with charts
✅ Excel export with embedded visualizations
✅ Mobile-responsive design
✅ Automatic data refresh

### 💡 Pro Tips
- Add custom domain (paid feature)
- Enable password protection via Streamlit secrets
- Monitor app analytics in Streamlit Cloud dashboard
- Set up email alerts for app errors
