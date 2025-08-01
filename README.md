# AI + React Screening Test - Feedback System

A complete feedback system with AI-powered emotion sentiment analysis, built with Flask backend and React frontend.

## 🚀 Features

### Backend (Flask)
- **User Management**: Sign up and authentication system
- **AI Emotion Analysis**: Advanced sentiment analysis using trained models
- **SQLite Database**: Persistent storage for users and feedback
- **RESTful API**: Clean API endpoints for frontend integration
- **CORS Support**: Cross-origin resource sharing enabled

### Frontend (React)
- **Responsive Design**: Modern, mobile-friendly UI using Tailwind CSS
- **User Authentication**: Complete signup/login flow
- **Admin Dashboard**: Comprehensive analytics and feedback management
- **Real-time Analysis**: Instant emotion detection for feedback
- **Protected Routes**: Secure access control

## 📁 Project Structure

```
datafloat/
├── backend/                 # Flask backend
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   ├── emotion_model_final.pkl    # Trained emotion model
│   ├── emotion_vectorizer_final.pkl # Text vectorizer
│   └── instance/           # SQLite database
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── contexts/       # React contexts
│   │   ├── services/       # API services
│   │   └── App.tsx         # Main app component
│   ├── package.json        # Node.js dependencies
│   └── README.md           # Frontend documentation
└── README.md               # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the backend server:
```bash
python app.py
```

The backend will run on `http://localhost:5001`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will run on `http://localhost:3000`

## 🎯 Usage

### User Flow
1. **Sign Up**: Create a new account at `/signup`
2. **Sign In**: Login with your credentials at `/login`
3. **Submit Feedback**: Rate your experience and provide detailed feedback
4. **View Results**: See the AI-detected emotion for your feedback

### Admin Flow
1. **Admin Login**: Use credentials `admin` / `admin123` (static admin account)
2. **Dashboard**: Access comprehensive analytics at `/admin`
3. **Analytics**: View emotion distribution and feedback summary

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status and available emotions |
| POST | `/signup` | User registration |
| POST | `/login` | User authentication (includes admin login) |
| POST | `/admin/login` | Dedicated admin authentication |
| POST | `/feedback` | Submit feedback with emotion analysis |
| GET | `/admin/feedbacks` | Get all feedbacks (admin only) |
| POST | `/emotion` | Analyze emotion of any text |

## 🧠 AI Model

The system uses a trained emotion classification model that can detect 13 different emotions:
- **Positive**: happiness, love, fun, enthusiasm, relief
- **Neutral**: neutral, surprise, empty, boredom
- **Negative**: sadness, worry, anger, hate

## 🎨 Technologies Used

### Backend
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **SQLite**: Database
- **scikit-learn**: Machine learning
- **NLTK**: Natural language processing
- **Flask-CORS**: Cross-origin support

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type safety
- **React Router**: Navigation
- **Axios**: HTTP client
- **Tailwind CSS**: Styling
- **Context API**: State management

## 📱 Responsive Design

The frontend is fully responsive and optimized for:
- Desktop computers (1200px+)
- Tablets (768px - 1199px)
- Mobile phones (320px - 767px)

## 🔒 Security Features

- Protected routes for authenticated users
- Static admin account (admin/admin123) for administration
- Admin-only access to dashboard
- Input validation and sanitization
- Secure password handling

## 🚀 Deployment

### Backend Deployment
1. Set up a production server
2. Install Python dependencies
3. Configure environment variables
4. Use a production WSGI server (e.g., Gunicorn)

### Frontend Deployment
1. Build the production version:
```bash
npm run build
```
2. Deploy the `build` folder to a web server
3. Configure API base URL for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is created for the AI + React Screening Test.

## 🆘 Support

For issues and questions:
1. Check the documentation in each directory
2. Review the API endpoints
3. Ensure both backend and frontend are running
4. Check browser console for errors

---

**Note**: Make sure both the backend and frontend are running simultaneously for the application to work properly. 