import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { feedbackAPI } from '../services/api';

const Feedback: React.FC = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [feedback, setFeedback] = useState('');
  const [rating, setRating] = useState(5);
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [emotion, setEmotion] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!feedback.trim()) return;

    setLoading(true);
    try {
      const response = await feedbackAPI.submit({
        user_id: user!.id,
        text: feedback,
      });
      setEmotion(response.emotion);
      setSubmitted(true);
    } catch (err: any) {
      console.error('Feedback submission failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  if (submitted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-gray-900 mb-4">
              Thank you for your feedback!
            </h2>
            <div className="bg-white p-6 rounded-lg shadow">
              <p className="text-gray-600 mb-4">Your feedback has been submitted successfully.</p>
              <div className="bg-blue-50 p-4 rounded-md">
                <p className="text-sm text-blue-800">
                  <strong>Detected Emotion:</strong> {emotion}
                </p>
              </div>
            </div>
            <div className="mt-6 space-y-3">
              <button
                onClick={() => {
                  setSubmitted(false);
                  setFeedback('');
                  setEmotion('');
                }}
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Submit Another Feedback
              </button>
              <button
                onClick={handleLogout}
                className="w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white shadow rounded-lg p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900">Feedback Form</h2>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">Welcome, {user?.name}</span>
              <button
                onClick={handleLogout}
                className="text-sm text-red-600 hover:text-red-500"
              >
                Logout
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
         

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Please share your detailed feedback
              </label>
              <textarea
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                rows={6}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Tell us about your experience, what you liked, what could be improved..."
                required
              />
            </div>

            <div className="bg-blue-50 p-4 rounded-md">
              <p className="text-sm text-blue-800">
                <strong>Note:</strong> Your feedback will be analyzed using AI to understand the emotional sentiment and help us improve our services.
              </p>
            </div>

            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => {
                  setFeedback('');
                  setRating(5);
                }}
                className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Clear Form
              </button>
              <button
                type="submit"
                disabled={loading || !feedback.trim()}
                className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
              >
                {loading ? 'Submitting...' : 'Submit Feedback'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Feedback; 