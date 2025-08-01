# Frontend - AI + React Screening Test

A responsive React frontend for the AI-powered feedback system with emotion sentiment analysis.

## Features

- **User Authentication**: Sign up and sign in functionality
- **Admin Panel**: Static admin login (admin/admin123) with dashboard
- **Feedback System**: User feedback submission with AI emotion analysis
- **Responsive Design**: Modern, mobile-friendly UI using Tailwind CSS
- **Real-time Analysis**: Emotion detection for submitted feedback

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Backend server running on `http://localhost:5001`

## Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`

## Usage

### User Flow
1. **Sign Up**: Create a new account with email and password
2. **Sign In**: Login with your credentials
3. **Submit Feedback**: Rate your experience and provide detailed feedback
4. **View Results**: See the AI-detected emotion for your feedback

### Admin Flow
1. **Admin Login**: Use credentials `admin` / `admin123`
2. **Dashboard**: View all feedbacks with emotion analysis
3. **Analytics**: See emotion distribution and summary statistics

## API Endpoints

The frontend connects to the following backend endpoints:
- `POST /signup` - User registration
- `POST /login` - User authentication
- `POST /feedback` - Submit feedback with emotion analysis
- `GET /admin/feedbacks` - Get all feedbacks (admin only)
- `POST /emotion` - Analyze emotion of text

## Technologies Used

- **React 18** with TypeScript
- **React Router** for navigation
- **Axios** for API communication
- **Tailwind CSS** for styling
- **Context API** for state management

## Project Structure

```
src/
├── components/          # React components
│   ├── Signup.tsx      # User registration
│   ├── Login.tsx       # User authentication
│   ├── Feedback.tsx    # Feedback submission
│   ├── AdminDashboard.tsx # Admin panel
│   └── ProtectedRoute.tsx # Route protection
├── contexts/           # React contexts
│   └── AuthContext.tsx # Authentication state
├── services/           # API services
│   └── api.ts         # API communication
└── App.tsx            # Main application
```

## Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

## Backend Integration

Make sure your Flask backend is running on `http://localhost:5001` with CORS enabled. The backend should have:

- SQLite database with User and Feedback tables
- Emotion analysis model loaded
- All required API endpoints implemented

## Responsive Design

The application is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones

All components use Tailwind CSS utility classes for consistent styling across devices.
