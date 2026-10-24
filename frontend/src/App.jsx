import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, ShieldCheck, AlertTriangle, Terminal, Image as ImageIcon, Video as VideoIcon, Loader2, Scan } from 'lucide-react';

// Configurable via .env (VITE_API_URL) so the frontend can point at a local
// ngrok tunnel during development or a permanently deployed backend (e.g. Render) in production.
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://pry-mandarin-unsoiled.ngrok-free.dev';

function App() {
  // --- Re-engineered Splash Screen Sequencer ---
  const [showSplash, setShowSplash] = useState(true);
  const [splashStep, setSplashStep] = useState(0);
  const [loadingProgress, setLoadingProgress] = useState(0);

  // --- Core State ---
  const [scanMode, setScanMode] = useState('image'); // 'image' | 'video'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const logsEndRef = useRef(null);

  // Bulletproof React Animation Sequencer
  useEffect(() => {
    // Step 1: Reveal Shield (0.1s)
    const t1 = setTimeout(() => setSplashStep(1), 100);
    // Step 2: Reveal Text (0.6s)
    const t2 = setTimeout(() => setSplashStep(2), 600);
    // Step 3: Reveal Loading Bar (1.2s)
    const t3 = setTimeout(() => setSplashStep(3), 1200);
    // Step 4: Trigger Fade Out (3.0s)
    const t4 = setTimeout(() => setSplashStep(4), 3000);
    // Step 5: Physically Unmount (3.5s)
    const t5 = setTimeout(() => setShowSplash(false), 3500);

    // Progress bar math
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 3;
      if (progress >= 100) {
        progress = 100;
        clearInterval(progressInterval);
      }
      setLoadingProgress(progress);
    }, 70);

    return () => {
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3);
      clearTimeout(t4); clearTimeout(t5); clearInterval(progressInterval);
    };
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleModeChange = (mode) => {
    if (mode === scanMode) return;
    setScanMode(mode);
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setLogs([]);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setLogs([]);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    const isVideo = scanMode === 'video';
    const endpoint = isVideo ? '/predict-video' : '/predict';
    setLogs([
      "Initiating secure connection to TrueLens API...",
      isVideo ? "Uploading video payload..." : "Uploading tensor payload...",
    ]);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const data = response.data;
      if (data.status === 'success') {
        setResult({
          prediction: data.prediction,
          confidence: data.confidence_percentage,
          raw: data.raw_score,
          framesAnalyzed: data.frames_analyzed,
          framesSkipped: data.frames_skipped,
          manipulatedFrameRatio: data.manipulated_frame_ratio,
        });
        setLogs(prev => [...prev, ...data.logs]);
      } else {
        setError(data.message || "An error occurred during analysis.");
        if (data.logs) setLogs(prev => [...prev, ...data.logs]);
      }
    } catch (err) {
      console.error(err);
      // Grab the REAL error from the backend instead of hardcoding it
      const errorMessage = err.response?.data?.detail || "Failed to connect to the backend server.";
      setError(errorMessage);
      setLogs(prev => [...prev, `CRITICAL ERROR: ${errorMessage}`]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-darkBase text-gray-200 font-sans selection:bg-blue-500/30 relative overflow-hidden w-full">

      {/* --- Re-engineered Cinematic Splash Screen --- */}
      {showSplash && (
        <div className={`fixed inset-0 z-50 flex flex-col items-center justify-center bg-darkBase transition-opacity duration-500 ease-in-out ${splashStep >= 4 ? 'opacity-0' : 'opacity-100'}`}>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[60%] h-[60%] bg-blue-900/10 blur-[150px] rounded-full pointer-events-none z-0"></div>

          <div className="flex flex-col items-center z-10">
            {/* Shield */}
            <div className={`mb-8 transition-all duration-700 ease-out ${splashStep >= 1 ? 'scale-100 opacity-100 blur-none' : 'scale-50 opacity-0 blur-md'}`}>
              <ShieldCheck size={120} className="text-blue-500 animate-pulse drop-shadow-[0_0_40px_rgba(59,130,246,0.7)]" />
            </div>

            {/* Text */}
            <h1 className={`text-5xl font-black text-white tracking-tighter transition-all duration-700 ease-out ${splashStep >= 2 ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0'}`}>
              TrueLens
            </h1>

            {/* Subtitle */}
            <p className={`text-blue-400 mt-4 uppercase text-xs font-semibold transition-all duration-700 ease-out ${splashStep >= 2 ? 'tracking-[0.3em] opacity-100' : 'tracking-[1em] opacity-0'}`}>
              Deepfake Forensics Engine
            </p>

            {/* Loading Bar */}
            <div className={`w-56 h-1 bg-gray-900 rounded-full mt-10 overflow-hidden relative border border-white/5 transition-all duration-700 ease-out ${splashStep >= 3 ? 'opacity-100' : 'opacity-0'}`}>
              <div className="absolute top-0 left-0 h-full bg-blue-500 rounded-full shadow-[0_0_15px_rgba(59,130,246,0.6)] transition-all duration-100 ease-out" style={{ width: `${loadingProgress}%` }} />
            </div>
          </div>
        </div>
      )}

      {/* --- Ambient Background Glow --- */}
      <div className="absolute top-[-5%] left-[-5%] w-[40%] h-[40%] bg-blue-900/20 blur-[130px] rounded-full pointer-events-none z-0"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[35%] h-[35%] bg-emerald-900/10 blur-[110px] rounded-full pointer-events-none z-0"></div>

      {/* --- Main App Interface --- */}
      <div className="relative z-10 w-full px-4 sm:px-8 md:px-12 py-10">
        <header className="max-w-[85rem] mx-auto mb-10 border-b border-white/10 pb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3 drop-shadow-md">
              <ShieldCheck className="text-blue-500" size={32} />
              TrueLens Engine
            </h1>
            <p className="text-gray-400 mt-2 font-light text-base">Architected by Harshada</p>
          </div>
          <div className="hidden md:flex items-center gap-2.5 text-sm text-gray-400 bg-white/5 backdrop-blur-md px-5 py-2.5 rounded-full border border-white/10 shadow-lg">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_10px_rgba(16,185,129,0.8)]"></div>
            System Online
          </div>
        </header>

        <main className="max-w-[85rem] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

          {/* Left Column: Upload */}
          <div className="lg:col-span-5 space-y-6 sticky top-10">

            {/* Mode Toggle */}
            <div className="flex bg-[#1a2235]/60 backdrop-blur-xl border border-white/10 rounded-xl p-1.5 gap-1.5">
              <button
                onClick={() => handleModeChange('image')}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold tracking-wide transition-all duration-300 ${scanMode === 'image' ? 'bg-blue-600 text-white shadow-[0_0_20px_rgba(79,70,229,0.4)]' : 'text-gray-400 hover:text-gray-200'}`}
              >
                <ImageIcon size={16} /> Image
              </button>
              <button
                onClick={() => handleModeChange('video')}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold tracking-wide transition-all duration-300 ${scanMode === 'video' ? 'bg-blue-600 text-white shadow-[0_0_20px_rgba(79,70,229,0.4)]' : 'text-gray-400 hover:text-gray-200'}`}
              >
                <VideoIcon size={16} /> Video
              </button>
            </div>

            <div onClick={() => fileInputRef.current?.click()} className="bg-[#1a2235]/60 backdrop-blur-xl border border-white/10 hover:border-blue-500/50 rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 group relative overflow-hidden shadow-2xl hover:shadow-[0_0_40px_rgba(59,130,246,0.15)] hover:-translate-y-1">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept={scanMode === 'video' ? 'video/mp4, video/quicktime, video/webm' : 'image/jpeg, image/png'}
                className="hidden"
              />

              {previewUrl ? (
                <div className="relative w-full aspect-square rounded-xl overflow-hidden border border-white/10 shadow-inner group-hover:border-blue-500/30 transition-all duration-300">
                  {scanMode === 'video' ? (
                    <video src={previewUrl} className="w-full h-full object-cover" muted loop autoPlay playsInline />
                  ) : (
                    <img src={previewUrl} alt="Target" className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
                  )}
                  {isLoading && (
                    <>
                      <div className="absolute inset-0 bg-blue-950/40 backdrop-blur-[2px]"></div>
                      {/* Fixed Scan line for standard tailwind */}
                      <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-b from-transparent via-blue-500 to-transparent shadow-[0_0_15px_rgba(59,130,246,0.8)] opacity-70 animate-pulse"></div>
                      <div className="absolute inset-0 flex flex-col gap-3 items-center justify-center">
                        <Scan size={64} className="text-blue-300 opacity-60 drop-shadow-[0_0_20px_rgba(59,130,246,0.5)] animate-pulse" />
                        <div className="absolute top-4 left-4 font-mono text-[10px] text-blue-300 bg-blue-950/70 p-1 rounded border border-blue-500/30">VECTORS_ACTIVE</div>
                      </div>
                    </>
                  )}
                  <div className="absolute inset-0 bg-darkBase/70 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-md">
                    <span className="bg-white/10 text-white px-5 py-2.5 rounded-xl border border-white/20 font-medium tracking-wide text-sm shadow-xl">Change Target</span>
                  </div>
                </div>
              ) : (
                <div className="py-16 flex flex-col items-center justify-center space-y-4">
                  <div className="bg-blue-500/10 p-5 rounded-xl group-hover:scale-110 transition-transform duration-300 border border-blue-500/20 shadow-inner">
                    <Upload className="text-blue-400 group-hover:text-blue-300" size={32} />
                  </div>
                  <div>
                    <p className="text-gray-100 font-medium text-lg tracking-tight">Initialize Target Scan</p>
                    <p className="text-gray-500 text-sm mt-2">Drag & Drop or Click to Browse</p>
                    <p className="text-gray-600 text-[11px] mt-1.5 font-mono">
                      {scanMode === 'video' ? 'MP4, MOV, WEBM / MAX 50MB' : 'JPG, PNG / MAX 10MB'}
                    </p>
                  </div>
                </div>
              )}
            </div>

            <button onClick={handleAnalyze} disabled={!selectedFile || isLoading} className={`w-full py-4 rounded-xl font-bold tracking-widest uppercase text-sm flex items-center justify-center gap-3 transition-all duration-300 ${!selectedFile ? 'bg-white/5 border border-white/10 text-gray-600 cursor-not-allowed' : isLoading ? 'bg-blue-600/50 text-blue-200 cursor-wait backdrop-blur-md border border-blue-500/30' : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-[0_0_30px_rgba(79,70,229,0.3)] hover:shadow-[0_0_40px_rgba(79,70,229,0.5)] border border-blue-400/30'}`}>
              {isLoading ? (
                <><Loader2 className="animate-spin" size={18} /> {scanMode === 'video' ? 'Sampling & Analyzing Frames...' : 'Neural Inference Active...'}</>
              ) : (
                <>{scanMode === 'video' ? <VideoIcon size={18} /> : <ImageIcon size={18} />} Execute Scan Pipeline</>
              )}
            </button>
          </div>

          {/* Right Column: Results & Terminal */}
          <div className="lg:col-span-7 flex flex-col space-y-6">

            <div className="bg-[#1a2235]/60 backdrop-blur-xl rounded-2xl border border-white/10 p-8 shadow-2xl min-h-[160px] flex flex-col justify-center relative overflow-hidden transition-all duration-300">
              {!result && !error && !isLoading && (
                <div className="text-center text-gray-500 space-y-3">
                  <ShieldCheck className="mx-auto opacity-20" size={48} />
                  <p className="font-light tracking-wide text-base">Awaiting input stream to initialize network.</p>
                </div>
              )}

              {isLoading && (
                <div className="text-center space-y-4">
                  <div className="text-blue-400 font-mono animate-pulse text-lg tracking-widest">[ PROCESSING TENSORS ]</div>
                  <div className="w-3/4 mx-auto bg-gray-900 rounded-full h-2 overflow-hidden border border-white/5 shadow-inner">
                    <div className="bg-gradient-to-r from-blue-600 via-indigo-400 to-blue-600 h-2 rounded-full animate-[loading_1.5s_ease-in-out_infinite] w-1/2 shadow-[0_0_15px_rgba(59,130,246,0.6)]"></div>
                  </div>
                </div>
              )}

              {error && (
                <div className="text-red-400 flex items-start gap-4 bg-red-950/20 backdrop-blur-md p-5 rounded-xl border border-red-900/50 shadow-inner">
                  <AlertTriangle size={24} className="shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-bold tracking-wide text-base mb-1">SYSTEM EXCEPTION</h3>
                    <p className="text-red-300/80 font-light text-sm">{error}</p>
                  </div>
                </div>
              )}

              {result && (
                <div className="z-10 transition-opacity duration-500">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-5">
                      {result.prediction === 'REAL' ? (
                        <div className="bg-emerald-500/10 p-4 rounded-xl border border-emerald-500/30 shadow-[0_0_30px_rgba(16,185,129,0.2)]">
                          <ShieldCheck className="text-emerald-400 drop-shadow-md" size={48} />
                        </div>
                      ) : (
                        <div className="bg-red-500/10 p-4 rounded-xl border border-red-500/30 shadow-[0_0_30px_rgba(239,68,68,0.2)]">
                          <AlertTriangle className="text-red-400 drop-shadow-md" size={48} />
                        </div>
                      )}
                      <div>
                        <p className="text-xs text-gray-400 uppercase tracking-widest mb-1 font-semibold">Final Verdict</p>
                        <h2 className={`text-4xl font-black tracking-tight drop-shadow-lg ${result.prediction === 'REAL' ? 'text-emerald-400' : 'text-red-500'}`}>
                          {result.prediction}
                        </h2>
                      </div>
                    </div>

                    <div className="text-right">
                      <p className="text-xs text-gray-400 uppercase tracking-widest mb-1 font-semibold">Confidence</p>
                      <p className="text-4xl font-light text-white tracking-tight">
                        {result.confidence}<span className="text-xl text-gray-500 font-normal">%</span>
                      </p>
                    </div>
                  </div>

                  {typeof result.framesAnalyzed === 'number' && (
                    <div className="mt-5 pt-4 border-t border-white/10 flex items-center gap-6 text-xs text-gray-400 font-mono">
                      <span>Frames analyzed: <span className="text-gray-200">{result.framesAnalyzed}</span></span>
                      <span>Frames skipped: <span className="text-gray-200">{result.framesSkipped}</span></span>
                      <span>Flagged ratio: <span className="text-gray-200">{Math.round((result.manipulatedFrameRatio || 0) * 100)}%</span></span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Terminal Console */}
            <div className="bg-[#090b10]/80 backdrop-blur-2xl rounded-2xl border border-white/5 overflow-hidden flex flex-col shadow-2xl relative transition-all duration-300 flex-grow min-h-[350px]">
              <div className="bg-white/5 px-5 py-3 border-b border-white/5 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Terminal size={14} className="text-gray-400" />
                  <span className="text-[11px] text-gray-400 font-mono tracking-widest uppercase">sys_forensics.log</span>
                </div>
                <div className="flex gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500/60 shadow-inner"></div>
                  <div className="w-3 h-3 rounded-full bg-yellow-500/60 shadow-inner"></div>
                  <div className="w-3 h-3 rounded-full bg-green-500/60 shadow-inner"></div>
                </div>
              </div>

              <div className="p-6 font-mono text-[13px] leading-relaxed h-[300px] overflow-y-auto space-y-3 custom-scrollbar flex-grow">
                <div className="text-gray-500 mb-6 border-b border-white/5 pb-4 font-light text-[12px]">
                  [ TrueLens Architecture v1.0.4 ]<br />
                  [ Backend: FastAPI / Machine Learning: EfficientNetB0 ]<br />
                  [ VPC_ISOLATION: ACTIVE / REGION: US-EAST-1 ]<br />
                  Awaiting tensor matrix...
                </div>

                {logs.map((log, index) => (
                  <div key={index} className={`flex transition-opacity duration-300 ${log.includes('CRITICAL') ? 'text-red-400 font-bold' : log.includes('REAL') ? 'text-emerald-400 font-bold' : log.includes('MANIPULATED') ? 'text-red-400 font-bold' : 'text-indigo-200/80'}`}>
                    <span className="text-indigo-500/50 mr-3 select-none">{`>`}</span>
                    <span className="break-words">{log}</span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>

          </div>
        </main>
      </div>
    </div>
  );
}

export default App;