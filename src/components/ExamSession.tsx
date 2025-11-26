import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { AlertTriangle, Eye, MessageSquare, Send, Camera, ShieldAlert, Clock, CheckCircle, Home, ArrowLeft, ArrowRight, Mic, MicOff } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { io } from 'socket.io-client';
import { Link, useLocation } from 'react-router-dom';
import { detectFace, cleanup } from '../lib/faceDetection';
import { detectEyeMetrics, cleanupEyeTracker } from '../lib/eyeTracking';
import type { GazeDirection } from '../lib/eyeTracking';
import { detectAIContent, checkPlagiarism } from '../lib/gemini';
import { GaussianDFTAudioDetectorFast } from '../lib/AudioDetectorFast';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const socket = io('http://localhost:3001');

interface Question {
  id: string;
  text: string;
  answer: string;
}

interface Anomaly {
  type: string;
  count: number;
  severity: 'low' | 'medium' | 'high';
  description: string;
  timestamp: Date;
}

interface AudioAnalysisResult {
  speechPercentage: number;
  totalFrames: number;
  duration: number;
  logLikelihood: number[];
  speechFrames: boolean[];
}

export function ExamSession() {
  const location = useLocation();
  const assessment = location.state?.assessment;
  const [questions, setQuestions] = useState<Question[]>(
    assessment?.questions.map((q: Question) => ({ ...q, answer: '' })) || []
  );
  
  const webcamRef = useRef<Webcam>(null);
  const activityListRef = useRef<HTMLDivElement>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [isAutomatedContentDetected, setIsAutomatedContentDetected] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [hasWebcamAccess, setHasWebcamAccess] = useState(false);
  const [timeLeft, setTimeLeft] = useState(assessment?.totalTime * 60 || 3600);
  const [examStarted, setExamStarted] = useState(false);
  const [examCompleted, setExamCompleted] = useState(false);
  const [examId] = useState('exam-123');
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [tabSwitchCount, setTabSwitchCount] = useState(0);
  const [outOfFrameCount, setOutOfFrameCount] = useState(0);
  const [phoneUsageCount, setPhoneUsageCount] = useState(0);
  const [multipleFacesCount, setMultipleFacesCount] = useState(0);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [submitAttempted, setSubmitAttempted] = useState(false);
  const [faceDetectionInterval, setFaceDetectionInterval] = useState<NodeJS.Timeout | null>(null);
  // Eye tracking states
  const [gazeDirection, setGazeDirection] = useState<GazeDirection | null>(null);
  const [blinkCount, setBlinkCount] = useState(0);
  const [prevBlinkLeft, setPrevBlinkLeft] = useState(false);
  const [prevBlinkRight, setPrevBlinkRight] = useState(false);
  const [awayConsecutiveSeconds, setAwayConsecutiveSeconds] = useState(0);
  const [gazeAwayEvents, setGazeAwayEvents] = useState(0);
  const [lastGazeAlertTime, setLastGazeAlertTime] = useState(0);
  const lastMultiFaceAlertRef = useRef(0);
  
  // Audio recording states
  const [audioRecorder, setAudioRecorder] = useState<MediaRecorder | null>(null);
  const [audioAnalysisResult, setAudioAnalysisResult] = useState<AudioAnalysisResult | null>(null);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden && examStarted && !examCompleted) {
        setTabSwitchCount(prev => prev + 1);
        setWarnings(prev => [...prev, 'Tab switching detected']);
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (faceDetectionInterval) {
        clearInterval(faceDetectionInterval);
      }
      cleanup();
      cleanupEyeTracker();
    };
  }, [examStarted, examCompleted, faceDetectionInterval]);

  // Keep Activity Monitor scrolled to the latest entry
  useEffect(() => {
    const el = activityListRef.current;
    if (!el) return;
    // Scroll to bottom on warnings update
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [warnings]);

  useEffect(() => {
    if (examStarted && timeLeft > 0 && !examCompleted) {
      const timer = setInterval(() => {
        setTimeLeft((prev) => prev - 1);
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [examStarted, timeLeft, examCompleted]);

  // Audio setup and recording functionality
  const setupAudioRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
          console.log('Audio chunk received:', event.data.size);
        }
      };
      
      recorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        
        // Store audio locally with better organization
        const reader = new FileReader();
        reader.onload = () => {
          const base64Audio = reader.result as string;
          const examId = Date.now().toString();
          const audioData = {
            id: examId,
            audioData: base64Audio,
            timestamp: new Date().toISOString(),
            duration: 0, // Will be updated after analysis
            size: audioBlob.size
          };
          
          // Store current exam audio
          localStorage.setItem('currentExamAudio', JSON.stringify(audioData));
          
          // Store in historical data
          const historicalAudio = JSON.parse(localStorage.getItem('examAudioHistory') || '[]');
          historicalAudio.unshift(audioData);
          // Keep only last 10 recordings
          if (historicalAudio.length > 10) {
            historicalAudio.splice(10);
          }
          localStorage.setItem('examAudioHistory', JSON.stringify(historicalAudio));
          
          console.log('Audio stored locally:', audioData.id);
        };
        reader.readAsDataURL(audioBlob);
        
        // Process audio for analysis
        processAudioForAnalysis(audioBlob);
      };
      
      recorder.start(1000); // Collect data every second
      setAudioRecorder(recorder);
      
      console.log('Audio recording started');
        } catch (error) {
      console.error('Error setting up audio recording:', error);
    }
  };

  useEffect(() => {
    socket.emit('join-exam', examId);

    socket.on('activity-alert', (data) => {
      if (!examCompleted) {
        setWarnings(prev => [...prev, data.message]);
      }
    });

    return () => {
      socket.off('activity-alert');
    };
  }, [examId, examCompleted]);

  useEffect(() => {
    if (hasWebcamAccess && examStarted && !examCompleted) {
      const interval = setInterval(async () => {
        if (webcamRef.current?.video && !examCompleted) {
          const videoEl = webcamRef.current.video as HTMLVideoElement;
          if (!videoEl || videoEl.readyState < 2 || videoEl.videoWidth === 0) {
            return;
          }
          const result = await detectFace(videoEl);
          
          if (!result.faceDetected && faceDetected) {
            setOutOfFrameCount(prev => prev + 1);
            socket.emit('suspicious-activity', {
              type: 'face-not-detected',
              message: 'No face detected in frame'
            });
            setFaceDetected(false);
          } else if (result.multipleFaces) {
            const now = Date.now();
            // Throttle multiple-faces alerts to once every 5 seconds
            if (now - lastMultiFaceAlertRef.current > 5000) {
              const faces = Array.isArray((result as any).predictions) ? (result as any).predictions.length : undefined;
              setMultipleFacesCount((c) => c + 1);
              setWarnings((prev) => [
                ...prev,
                `Multiple faces detected${faces && faces > 1 ? ` (${faces})` : ''}`
              ]);
              socket.emit('suspicious-activity', {
                type: 'multiple-faces',
                message: `Multiple faces detected${faces && faces > 1 ? ` (${faces})` : ''}`
              });
              lastMultiFaceAlertRef.current = now;
            }
          }

          if (result.phoneDetected) {
            setPhoneUsageCount(prev => prev + 1);
            socket.emit('suspicious-activity', {
              type: 'phone-detected',
              message: 'Phone usage detected'
            });
            setWarnings(prev => [...prev, 'Phone usage detected']);
          }
          
          setFaceDetected(result.faceDetected);

          // Eye tracking (blink/gaze) sampling with smoothing
          const eye = await detectEyeMetrics(videoEl);
          if (eye) {
            setGazeDirection(eye.gaze);
            // Blink detection: count on rising edge per eye
            if ((eye.blinkLeft && !prevBlinkLeft) || (eye.blinkRight && !prevBlinkRight)) {
              setBlinkCount((c) => c + 1);
            }
            setPrevBlinkLeft(eye.blinkLeft);
            setPrevBlinkRight(eye.blinkRight);

            const isAway = eye.gaze === 'left' || eye.gaze === 'right' || eye.gaze === 'down';
            setAwayConsecutiveSeconds((sec) => {
              const next = isAway ? sec + 1 : 0;
              // Alert if sustained away for >= 2s and last alert > 10s ago
              if (isAway && next >= 2 && Date.now() - lastGazeAlertTime > 10000) {
                setLastGazeAlertTime(Date.now());
                setGazeAwayEvents((e) => e + 1);
                socket.emit('suspicious-activity', {
                  type: 'gaze-away',
                  message: `Sustained gaze away: ${eye.gaze}`
                });
                setWarnings((prev) => [...prev, `Sustained gaze away: ${eye.gaze}`]);
              }
              return next;
            });
          }
        }
      }, 1000);

      setFaceDetectionInterval(interval);
      return () => {
        clearInterval(interval);
        setFaceDetectionInterval(null);
      };
    }
  }, [faceDetected, hasWebcamAccess, examStarted, examCompleted]);

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const handleTextChange = async (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    const updatedQuestions = [...questions];
    updatedQuestions[currentQuestionIndex].answer = text;
    setQuestions(updatedQuestions);
    
    if (text.length > 30 && !examCompleted) {
      const [aiResult, plagiarismResult] = await Promise.all([
        detectAIContent(text),
        checkPlagiarism(text)
      ]);

      if (aiResult) {
        setIsAutomatedContentDetected(true);
        socket.emit('suspicious-activity', {
          type: 'content-detection',
          message: 'Potential automated content detected'
        });
        setWarnings(prev => [...prev, 'Automated content detected']);
      }

      if (plagiarismResult.isPlagiarized) {
        socket.emit('suspicious-activity', {
          type: 'plagiarism',
          message: `Potential plagiarism detected (${Math.round(plagiarismResult.similarity * 100)}% similarity)`
        });
        setWarnings(prev => [...prev, `Plagiarism detected (${Math.round(plagiarismResult.similarity * 100)}% similarity)`]);
      }
    }
  };

  const processAudioForAnalysis = async (audioBlob: Blob) => {
    try {
      const audioBuffer = await audioBlob.arrayBuffer();
      
      // Initialize audio detector with balanced settings
      const detector = new GaussianDFTAudioDetectorFast(
        1024,     // frameLength
        512,      // hopLength  
        16000,    // sampleRate
        0.003,    // minEnergyThreshold (reduced from 0.01 to 0.003)
        80,       // minFrequency (human speech range)
        3400      // maxFrequency (human speech range)
      );
      
      // Analyze the entire audio buffer with moderate threshold
      const result = await detector.detectSpeechFromArrayBuffer(audioBuffer, 0.3); // Reduced from 0.7 to 0.3
      
      // Create analysis result
      const analysisResult = {
        speechPercentage: result.speechPercentage,
        totalFrames: result.totalFrames,
        speechFrames: result.speechFrames, // Keep as boolean array
        duration: result.duration,
        logLikelihood: result.logLikelihood
      };
      
      setAudioAnalysisResult(analysisResult);
      
      // Store analysis results locally with updated duration
      localStorage.setItem('examAudioAnalysis', JSON.stringify(analysisResult));
      
      // Update stored audio data with duration
      const currentAudioData = JSON.parse(localStorage.getItem('currentExamAudio') || '{}');
      if (currentAudioData.id) {
        currentAudioData.duration = result.duration;
        localStorage.setItem('currentExamAudio', JSON.stringify(currentAudioData));
      }
      
      console.log('Audio analysis completed:', {
        speechPercentage: analysisResult.speechPercentage.toFixed(2) + '%',
        speechFrames: analysisResult.speechFrames.filter(Boolean).length,
        totalFrames: analysisResult.totalFrames,
        duration: analysisResult.duration.toFixed(2) + 's',
        avgLogLikelihood: (result.logLikelihood.reduce((a, b) => a + b, 0) / result.logLikelihood.length).toFixed(3),
        thresholds: 'Energy: 0.003, Frequency: 80-3400Hz, Detection: 30%'
      });
    } catch (error) {
      console.error('Error processing audio:', error);
      // Set a default result in case of error
      const defaultResult = {
        speechPercentage: 0,
        totalFrames: 0,
        speechFrames: [],
        duration: 0,
        logLikelihood: []
      };
      setAudioAnalysisResult(defaultResult);
    }
  };

  const handleWebcamAccess = (stream: MediaStream | null) => {
    setWebcamStream(stream);
    setHasWebcamAccess(!!stream);
    if (stream) {
      setExamStarted(true);
      // Setup audio recording when webcam access is granted
      setupAudioRecording();
    }
  };

  const handleSubmit = () => {
    const hasEmptyAnswers = questions.some(q => !q.answer.trim());
    if (hasEmptyAnswers) {
      setSubmitAttempted(true);
      setWarnings(prev => [...prev, 'Please answer all questions before submitting']);
      return;
    }

    setExamCompleted(true);
    
    // Stop audio recording
    if (audioRecorder && audioRecorder.state === 'recording') {
      audioRecorder.stop();
    }
    
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      setWebcamStream(null);
    }
    if (faceDetectionInterval) {
      clearInterval(faceDetectionInterval);
      setFaceDetectionInterval(null);
    }
    cleanup();

    let contentAnomaly: Anomaly | undefined;
    if (isAutomatedContentDetected) {
      contentAnomaly = {
        type: 'Content Detection',
        count: 1,
        severity: 'high',
        description: 'Potential use of automated content',
        timestamp: new Date()
      };
    }

    // Add audio anomaly if speech detected
    let audioAnomaly: Anomaly | undefined;
    if (audioAnalysisResult && audioAnalysisResult.speechPercentage > 5) { // Reduced from 20% to 5%
      audioAnomaly = {
        type: 'Speech Detection',
        count: audioAnalysisResult.speechFrames.filter(Boolean).length,
        severity: audioAnalysisResult.speechPercentage > 25 ? 'high' : audioAnalysisResult.speechPercentage > 10 ? 'medium' : 'low', // Adjusted thresholds
        description: `Speech detected in ${audioAnalysisResult.speechPercentage.toFixed(1)}% of exam time`,
        timestamp: new Date()
      };
    }

    if (tabSwitchCount > 0 || isAutomatedContentDetected || (audioAnalysisResult && audioAnalysisResult.speechPercentage > 5)) { // Reduced from 20% to 5%
      const activity = {
        id: Date.now().toString(),
        type: 'High Risk Activity Detected',
        description: [
          tabSwitchCount > 0 ? `Tab switching detected (${tabSwitchCount} times).` : '',
          isAutomatedContentDetected ? 'Automated content detected.' : '',
          audioAnalysisResult && audioAnalysisResult.speechPercentage > 5 ? `Speech detected (${audioAnalysisResult.speechPercentage.toFixed(1)}%).` : '', // Reduced threshold
          multipleFacesCount > 0 ? `Multiple faces detected (${multipleFacesCount} time${multipleFacesCount > 1 ? 's' : ''}).` : ''
        ].filter(Boolean).join(' '),
        timestamp: new Date(),
        severity: 'high'
      };
      const existingActivities = JSON.parse(localStorage.getItem('recentActivities') || '[]');
      localStorage.setItem('recentActivities', JSON.stringify([activity, ...existingActivities.slice(0, 9)]));
    }
    
    const finalAnomalies: Anomaly[] = [
      {
        type: 'Tab Switching',
        count: tabSwitchCount,
        severity: tabSwitchCount > 5 ? 'high' : tabSwitchCount > 2 ? 'medium' : 'low',
        description: 'Switched between browser tabs during exam',
        timestamp: new Date()
      },
      {
        type: 'Face Detection',
        count: outOfFrameCount,
        severity: outOfFrameCount > 10 ? 'high' : outOfFrameCount > 5 ? 'medium' : 'low',
        description: 'Face not detected in camera frame',
        timestamp: new Date()
      },
      {
        type: 'Gaze Away',
        count: gazeAwayEvents,
        severity: gazeAwayEvents > 5 ? 'high' : gazeAwayEvents > 2 ? 'medium' : 'low',
        description: 'Repeated sustained gaze away from screen',
        timestamp: new Date()
      },
      {
        type: 'Multiple Faces',
        count: multipleFacesCount,
        severity: multipleFacesCount > 3 ? 'high' : multipleFacesCount > 1 ? 'medium' : 'low',
        description: 'Multiple faces detected in camera frame',
        timestamp: new Date()
      },
  ...(contentAnomaly ? [contentAnomaly] : []),
      ...(audioAnomaly ? [audioAnomaly] : [])
    ];
    setAnomalies(finalAnomalies);

    const examResult = {
      id: Date.now().toString(),
      timestamp: new Date(),
      anomalies: finalAnomalies,
      warnings: warnings,
      tabSwitches: tabSwitchCount,
  aiDetected: isAutomatedContentDetected,
      phoneUsage: phoneUsageCount,
      outOfFrame: outOfFrameCount,
      audioAnalysis: audioAnalysisResult,
      gazeAwayEvents,
      blinkCount
    };
    const existingResults = JSON.parse(localStorage.getItem('examResults') || '[]');
    localStorage.setItem('examResults', JSON.stringify([examResult, ...existingResults]));
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
    }
  };

  const handlePrevQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1);
    }
  };

  if (examCompleted) {
    // Create audio chart data
    const createAudioChartData = () => {
      if (!audioAnalysisResult) return null;
      
      return {
        labels: Array.from({ length: audioAnalysisResult.logLikelihood.length }, (_, i) => `Frame ${i + 1}`),
        datasets: [
          {
            label: 'Speech Likelihood',
            data: audioAnalysisResult.logLikelihood,
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.1,
            fill: true
          },
          {
            label: 'Speech Detection',
            data: audioAnalysisResult.speechFrames.map(v => v ? 1 : -1),
            borderColor: 'rgb(239, 68, 68)',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            tension: 0,
            stepped: true
          }
        ]
      };
    };

    const audioChartData = createAudioChartData();
    
    const chartOptions = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'Speech Detection Analysis'
        }
      },
      scales: {
        y: {
          min: -2,
          max: 2,
          title: {
            display: true,
            text: 'Likelihood Score'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Audio Frames'
          }
        }
      }
    };

    return (
      <motion.div 
        className="min-h-screen bg-amber-50 p-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="max-w-4xl mx-auto space-y-8">
          <motion.div 
            className="bg-white rounded-3xl shadow-xl p-8 border border-purple-100"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
          >
            <div className="text-center mb-8">
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-900">Exam Completed</h2>
              <p className="text-gray-600 mt-2">Your answers have been submitted successfully</p>
            </div>

            <div className="space-y-6">
              <h3 className="text-xl font-semibold text-gray-900 flex items-center space-x-2">
                <ShieldAlert className="w-6 h-6 text-purple-500" />
                <span>Proctor Report</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {anomalies.map((anomaly, index) => (
                  <motion.div
                    key={index}
                    className={`p-6 rounded-2xl border ${
                      anomaly.severity === 'high' 
                        ? 'bg-red-50 border-red-100' 
                        : anomaly.severity === 'medium'
                        ? 'bg-yellow-50 border-yellow-100'
                        : 'bg-green-50 border-green-100'
                    }`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <h4 className="font-semibold text-gray-900">{anomaly.type}</h4>
                        <p className="text-sm text-gray-600 mt-1">{anomaly.description}</p>
                      </div>
                      <span className={`text-2xl font-bold ${
                        anomaly.severity === 'high' 
                          ? 'text-red-600' 
                          : anomaly.severity === 'medium'
                          ? 'text-yellow-600'
                          : 'text-green-600'
                      }`}>
                        {anomaly.count}
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Audio Analysis Section */}
              {audioAnalysisResult && (
                <div className="mt-8 p-6 bg-blue-50 rounded-2xl">
                  <h4 className="font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                    <Mic className="w-5 h-5 text-blue-600" />
                    <span>Audio Analysis Results</span>
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="bg-white rounded-lg p-4">
                      <p className="text-sm text-gray-600">Speech Detected</p>
                      <p className="text-2xl font-bold text-blue-600">
                        {audioAnalysisResult.speechPercentage.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-white rounded-lg p-4">
                      <p className="text-sm text-gray-600">Speech Frames</p>
                      <p className="text-2xl font-bold text-green-600">
                        {audioAnalysisResult.speechFrames.filter(Boolean).length}
                      </p>
                    </div>
                    <div className="bg-white rounded-lg p-4">
                      <p className="text-sm text-gray-600">Audio Duration</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {audioAnalysisResult.duration.toFixed(1)}s
                      </p>
                    </div>
                  </div>
                  
                  {/* Audio Chart */}
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-4">Speech Detection Timeline</h5>
                    {audioChartData && (
                      <div style={{ height: '300px' }}>
                        <Line data={audioChartData} options={chartOptions} />
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="mt-8 p-6 bg-gray-100 rounded-2xl">
                <h4 className="font-semibold text-gray-900 mb-4">Suspicious Activities Timeline</h4>
                <div className="space-y-4">
                  {warnings.map((warning, index) => (
                    <motion.div
                      key={index}
                      className="flex items-center space-x-3 text-gray-700"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <AlertTriangle className="w-4 h-4 text-yellow-500" />
                      <span>{warning}</span>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>

            <motion.div 
              className="mt-8 flex justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              <Link 
                to="/dashboard"
                className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl font-semibold flex items-center space-x-3 shadow-lg shadow-blue-200/50 hover:shadow-xl hover:scale-105 transition-all duration-200"
              >
                <Home className="w-5 h-5" />
                <span>Return to Dashboard</span>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </motion.div>
    );
  }

  if (!hasWebcamAccess) {
    return (
      <motion.div 
        className="min-h-[80vh] flex flex-col items-center justify-center space-y-6 bg-amber-50"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Camera className="w-16 h-16 text-blue-500" />
        <h2 className="text-2xl font-bold text-gray-900">Camera Access Required</h2>
        <p className="text-gray-600 text-center max-w-md">
          To ensure exam integrity, we need access to your camera for proctoring.
          Please allow camera access to continue.
        </p>
        <Webcam
          ref={webcamRef}
          onUserMedia={handleWebcamAccess}
          className="hidden"
        />
      </motion.div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-4">
        <motion.div 
          className="bg-white rounded-3xl shadow-xl p-8 border border-gray-200"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900">
              Question {currentQuestionIndex + 1} of {questions.length}
            </h2>
            <motion.div 
              className="px-6 py-3 bg-gray-100 rounded-full flex items-center space-x-3"
              animate={{ scale: timeLeft <= 300 ? [1, 1.1, 1] : 1 }}
              transition={{ repeat: timeLeft <= 300 ? Infinity : 0, duration: 1 }}
            >
              <Clock className={`w-5 h-5 ${timeLeft <= 300 ? 'text-red-500' : 'text-blue-600'}`} />
              <span className={`font-semibold ${timeLeft <= 300 ? 'text-red-500' : 'text-blue-600'}`}>
                {formatTime(timeLeft)}
              </span>
            </motion.div>
          </div>
          <div className="prose max-w-none">
            <p className="text-gray-700 text-lg">
              {questions[currentQuestionIndex].text}
            </p>
          </div>
          <div className="mt-8 space-y-6">
            <div className="relative">
              <textarea
                className={`w-full h-64 p-6 border ${
                  submitAttempted && !questions[currentQuestionIndex].answer.trim()
                    ? 'border-red-300 focus:ring-red-400'
                    : 'border-indigo-200 focus:ring-purple-400'
                } rounded-2xl focus:ring-2 focus:border-transparent resize-none text-gray-700`}
                placeholder="Type your answer here..."
                value={questions[currentQuestionIndex].answer}
                onChange={handleTextChange}
              />
              {submitAttempted && !questions[currentQuestionIndex].answer.trim() && (
                <p className="text-red-500 text-sm mt-2">This question requires an answer</p>
              )}
            </div>
            
            <div className="flex items-center justify-between space-x-4">
              <motion.button
                className="px-6 py-4 bg-gray-100 text-gray-600 rounded-2xl font-semibold flex items-center space-x-2 disabled:opacity-50"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handlePrevQuestion}
                disabled={currentQuestionIndex === 0}
              >
                <ArrowLeft className="w-5 h-5" />
                <span>Previous</span>
              </motion.button>

              {currentQuestionIndex === questions.length - 1 ? (
                <motion.button
                  className="flex-1 py-4 bg-green-600 hover:bg-green-700 text-white rounded-2xl font-semibold flex items-center justify-center space-x-3 shadow"
                  whileHover={{ scale: 1.02, boxShadow: '0 20px 25px -5px rgb(99 102 241 / 0.2)' }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleSubmit}
                >
                  <Send className="w-5 h-5" />
                  <span>Submit All Answers</span>
                </motion.button>
              ) : (
                <motion.button
                  className="flex-1 py-4 bg-gray-900 hover:bg-black text-white rounded-2xl font-semibold flex items-center justify-center space-x-3 shadow"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleNextQuestion}
                >
                  <span>Next Question</span>
                  <ArrowRight className="w-5 h-5" />
                </motion.button>
              )}
            </div>

            <div className="flex justify-center space-x-2">
              {questions.map((_, index) => (
                <motion.div
                  key={index}
                  className={`w-3 h-3 rounded-full ${
                    index === currentQuestionIndex
                      ? 'bg-indigo-600'
                      : questions[index].answer.trim()
                      ? 'bg-green-400'
                      : 'bg-gray-300'
                  }`}
                  whileHover={{ scale: 1.2 }}
                  onClick={() => setCurrentQuestionIndex(index)}
                  style={{ cursor: 'pointer' }}
                />
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      <div className="space-y-4">
        <motion.div
          className="bg-white rounded-3xl shadow-xl overflow-hidden border border-gray-200"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="p-4 bg-gray-900">
            <h3 className="text-white font-semibold flex items-center space-x-2">
              <Eye className="w-5 h-5" />
              <span>Proctor View</span>
            </h3>
          </div>
          <div className="p-4">
            <Webcam
              ref={webcamRef}
              className="w-full rounded-2xl"
              mirrored
            />
          </div>
        </motion.div>

        <motion.div
          className="bg-white rounded-3xl shadow-xl border border-red-100"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="p-4 border-b border-gray-100">
            <h3 className="font-semibold text-gray-900 flex items-center space-x-2">
              <ShieldAlert className="w-5 h-5 text-red-500" />
              <span>Activity Monitor</span>
            </h3>
          </div>
          <div ref={activityListRef} className="p-4 space-y-4 max-h-[300px] overflow-y-auto">
            <AnimatePresence>
              {!faceDetected && (
                <motion.div
                  className="flex items-center space-x-2 text-red-600 bg-red-50 p-4 rounded-2xl"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                >
                  <AlertTriangle className="w-4 h-4" />
                  <span>Face not detected</span>
                </motion.div>
              )}
              
              {/* Audio Recording Status */}
              <motion.div
                className={`flex items-center space-x-2 p-4 rounded-2xl ${
                  audioRecorder 
                    ? 'text-green-600 bg-green-50' 
                    : 'text-yellow-600 bg-yellow-50'
                }`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                {audioRecorder ? (
                  <Mic className="w-4 h-4" />
                ) : (
                  <MicOff className="w-4 h-4" />
                )}
                <span>
                  {audioRecorder ? 'Audio monitoring active' : 'Audio monitoring inactive'}
                </span>
              </motion.div>
              
              {warnings.map((warning, index) => (
                <motion.div
                  key={index}
                  className="flex items-center space-x-2 text-amber-600 bg-amber-50 p-4 rounded-2xl"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <AlertTriangle className="w-4 h-4" />
                  <span>{warning}</span>
                </motion.div>
              ))}
              
              {isAutomatedContentDetected && (
                <motion.div
                  className="flex items-center space-x-2 text-red-600 bg-red-50 p-4 rounded-2xl"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <MessageSquare className="w-4 h-4" />
                  <span>Potential automated content detected</span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Eye Tracking Status */}
        <motion.div
          className="bg-white rounded-3xl shadow-xl border border-gray-200"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="p-4 border-b border-gray-100">
            <h3 className="font-semibold text-gray-900 flex items-center space-x-2">
              <Eye className="w-5 h-5 text-blue-600" />
              <span>Eye Tracking</span>
            </h3>
          </div>
          <div className="p-4 grid grid-cols-3 gap-4 text-sm">
            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <p className="text-gray-500">Gaze</p>
              <p className="font-semibold text-gray-900">{gazeDirection ?? '—'}</p>
            </div>
            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <p className="text-gray-500">Blinks</p>
              <p className="font-semibold text-gray-900">{blinkCount}</p>
            </div>
            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <p className="text-gray-500">Away streak</p>
              <p className="font-semibold text-gray-900">{awayConsecutiveSeconds}s</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          className={`${faceDetected ? 'bg-green-500' : 'bg-red-500'} rounded-3xl shadow-xl p-4 text-white`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="flex items-center space-x-2">
            <Eye className="w-5 h-5" />
            <span>IntigrityGuard {faceDetected ? 'Active' : 'Warning'}</span>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
