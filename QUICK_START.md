# 🚀 Quick Start Guide

## Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

## One-Command Setup

### Option 1: Automatic Start (Recommended)
```bash
./start.sh
```

### Option 2: Manual Start
```bash
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend
cd frontend
npm start
```

## Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5001

## Quick Test

1. **Visit**: http://localhost:3000
2. **Sign Up**: Create a new account
3. **Login**: Use your credentials
4. **Submit Feedback**: Rate and describe your experience
5. **View Results**: See AI-detected emotion

## Admin Access

- **Username**: admin
- **Password**: admin123
- **Dashboard**: http://localhost:3000/admin
- **Features**: View all feedbacks with emotion analysis
- **Note**: This is a static admin account for administration purposes

## Troubleshooting

### Frontend Issues
```bash
cd frontend
npm install
npm start
```

### Backend Issues
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Port Conflicts
- Backend uses port 5001
- Frontend uses port 3000
- Make sure these ports are available

## Features Available

✅ **User Authentication**
✅ **Feedback System** 
✅ **AI Emotion Analysis**
✅ **Admin Dashboard**
✅ **Responsive Design**
✅ **Real-time Analysis**

## Support

If you encounter any issues:
1. Check both terminals for error messages
2. Ensure all dependencies are installed
3. Verify ports are not in use
4. Check browser console for frontend errors 