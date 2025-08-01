import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { feedbackAPI, Feedback } from '../services/api';

const AdminDashboard: React.FC = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [feedbacks, setFeedbacks] = useState<Feedback[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchFeedbacks();
  }, []);

  const fetchFeedbacks = async () => {
    try {
      const data = await feedbackAPI.getAll();
      setFeedbacks(data);
    } catch (err: any) {
      setError('Failed to fetch feedbacks');
      console.error('Error fetching feedbacks:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getEmotionColor = (emotion: string) => {
    const colors: { [key: string]: string } = {
      happiness: 'bg-green-100 text-green-800',
      love: 'bg-pink-100 text-pink-800',
      fun: 'bg-yellow-100 text-yellow-800',
      enthusiasm: 'bg-orange-100 text-orange-800',
      relief: 'bg-blue-100 text-blue-800',
      neutral: 'bg-gray-100 text-gray-800',
      surprise: 'bg-purple-100 text-purple-800',
      sadness: 'bg-blue-100 text-blue-800',
      worry: 'bg-yellow-100 text-yellow-800',
      anger: 'bg-red-100 text-red-800',
      hate: 'bg-red-100 text-red-800',
      boredom: 'bg-gray-100 text-gray-800',
      empty: 'bg-gray-100 text-gray-800',
    };
    return colors[emotion] || 'bg-gray-100 text-gray-800';
  };

  const getEmotionStats = () => {
    const stats: { [key: string]: number } = {};
    feedbacks.forEach(feedback => {
      stats[feedback.emotion] = (stats[feedback.emotion] || 0) + 1;
    });
    return stats;
  };

  const emotionStats = getEmotionStats();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
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
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900">Total Feedbacks</h3>
            <p className="text-3xl font-bold text-indigo-600">{feedbacks.length}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900">Positive Emotions</h3>
            <p className="text-3xl font-bold text-green-600">
              {feedbacks.filter(f => ['happiness', 'love', 'fun', 'enthusiasm', 'relief'].includes(f.emotion)).length}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900">Neutral Emotions</h3>
            <p className="text-3xl font-bold text-gray-600">
              {feedbacks.filter(f => ['neutral', 'surprise', 'empty', 'boredom'].includes(f.emotion)).length}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900">Negative Emotions</h3>
            <p className="text-3xl font-bold text-red-600">
              {feedbacks.filter(f => ['sadness', 'worry', 'anger', 'hate'].includes(f.emotion)).length}
            </p>
          </div>
        </div>

        {/* Emotion Distribution */}
        <div className="bg-white rounded-lg shadow mb-8">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Emotion Distribution</h3>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {Object.entries(emotionStats).map(([emotion, count]) => (
                <div key={emotion} className="text-center">
                  <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getEmotionColor(emotion)}`}>
                    {emotion}
                  </div>
                  <p className="mt-2 text-2xl font-bold text-gray-900">{count}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Feedback List */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">All Feedbacks</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    User ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Feedback
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Emotion
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {feedbacks.map((feedback) => (
                  <tr key={feedback.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {feedback.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {feedback.user_id}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900">
                      <div className="max-w-xs truncate" title={feedback.text}>
                        {feedback.text}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getEmotionColor(feedback.emotion)}`}>
                        {feedback.emotion}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard; 