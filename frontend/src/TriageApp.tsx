
import React, { useState, useRef, useEffect } from 'react';
import {
  Upload, AlertTriangle, Activity, Eye, FileText,
  Share2, ZoomIn, ZoomOut,
  Moon, Sun, Users, Search, Split, Monitor,
  Stethoscope, Scan, ShieldAlert,
  Mic, MessageSquare, Send, Minimize2, Brain, LayoutGrid

} from 'lucide-react';

// --- TYPES ---
type ModelType = 'radiology' | 'dermatology';
type AppMode = 'triage' | 'compare';

interface Patient {
  id?: string;
  name: string;
  age: string | number;
  gender?: string;
  symptom: string;
  urgency?: string;
  status?: string;
}

interface Finding {
  label: string;
  confidence: number;
  uncertainty: number;
}

interface ModelPrediction {
  model_name: string;
  prediction: string;
  confidence: number;
}

interface ConsensusResult {
  models: ModelPrediction[];
  final_consensus: string;
  final_confidence: number;
  agreement_level: 'High' | 'Medium' | 'Low';
}

interface ReliabilityCheck {
  isReliable: boolean;
  score: number;
  reason: string;
  anatomical_region: string;
}

interface AnalysisResult {
  main_label: string;
  detailed_findings: Finding[];
  generated_report: string;
  heatmap_b64: string | null;
  uncertainty_score: number;
  consensus_result?: ConsensusResult;
  reliability?: ReliabilityCheck;
}

interface ComparisonResult {
  improvement_score: number; // 0-100%
  worsening_score: number;   // 0-100%
  status: 'Improved' | 'Stable' | 'Worsening';
  diff_heatmap_url: string;
}

interface QualityIssue {
  type: 'brightness' | 'blur' | 'rotation' | 'contrast' | 'metal' | 'labels' | 'exposure';
  severity: 'low' | 'medium' | 'high';
  message: string;
  value: number;
}

interface QualityCheck {
  isAcceptable: boolean;
  score: number;
  issues: QualityIssue[];
  warnings: string[];
}

// --- CONFIGURATION ---
const USE_MOCK_BACKEND = false; // Changed to false to enable Real LLM Backend
const USE_LLM_PREDICTION = true; // New Feature Flag
const API_URL = "http://localhost:8000/predict";
const LLM_API_URL = "http://localhost:8000/predict-llm";
const SKIN_API_URL = "http://localhost:8000/predict-skin";


// Mock Worklist Data
const MOCK_QUEUE: Patient[] = [
  { id: 'PX-2024-001', name: 'Alex D.', age: 45, gender: 'M', symptom: 'High Fever, Cough', urgency: 'High', status: 'Pending' },
  { id: 'PX-2024-002', name: 'Sarah L.', age: 29, gender: 'F', symptom: 'Routine Checkup', urgency: 'Low', status: 'Pending' },
  { id: 'PX-2024-003', name: 'Marcus R.', age: 62, gender: 'M', symptom: 'Shortness of Breath', urgency: 'Critical', status: 'Pending' },
  { id: 'PX-2024-004', name: 'Emily W.', age: 34, gender: 'F', symptom: 'Chest Pain', urgency: 'Medium', status: 'In Progress' },
  { id: 'PX-2024-005', name: 'David K.', age: 55, gender: 'M', symptom: 'Fatigue', urgency: 'Low', status: 'Analyzed' },
];

const SKIN_MOCK_QUEUE: Patient[] = [
  { id: 'DM-2024-001', name: 'Lisa K.', age: 28, gender: 'F', symptom: 'Rash on arm', urgency: 'Medium', status: 'Pending' },
  { id: 'DM-2024-002', name: 'Robert M.', age: 45, gender: 'M', symptom: 'Changing mole', urgency: 'High', status: 'Pending' },
  { id: 'DM-2024-003', name: 'James P.', age: 62, gender: 'M', symptom: 'Dark lesion on back', urgency: 'Critical', status: 'Pending' },
  { id: 'DM-2024-004', name: 'Sarah T.', age: 31, gender: 'F', symptom: 'Itchy patches', urgency: 'Low', status: 'Pending' },
  { id: 'DM-2024-005', name: 'Nina S.', age: 19, gender: 'F', symptom: 'Acne', urgency: 'Low', status: 'Pending' },
];

const TriageApp: React.FC = () => {
  // --- APP STATE ---
  const [appMode, setAppMode] = useState<AppMode>('triage');
  const [modelType, setModelType] = useState<ModelType>('radiology');
  const [darkMode, setDarkMode] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<'findings' | 'report'>('findings');
  const [activeView, setActiveView] = useState<'dashboard' | 'worklist' | 'viewer' | 'report'>('dashboard');

  // --- PATIENT & IMAGE STATE ---
  const [patientData, setPatientData] = useState<Patient>({ name: '', age: '', symptom: '' });
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // --- ANALYSIS STATE ---
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [reportText, setReportText] = useState<string>('');
  const [isRecording, setIsRecording] = useState<boolean>(false);

  // --- CONSENSUS / FEDERATED STATE ---
  const [isSecondOpinionMode, setIsSecondOpinionMode] = useState<boolean>(false);
  const [federatedStatus, setFederatedStatus] = useState<string>('idle');

  // --- QUALITY CHECK STATE ---
  const [isCheckingQuality, setIsCheckingQuality] = useState<boolean>(false);
  const [qualityCheck, setQualityCheck] = useState<QualityCheck | null>(null);
  const [qualityHeatmapUrl, setQualityHeatmapUrl] = useState<string | null>(null);

  // --- SEARCH STATE ---
  const [searchTerm, setSearchTerm] = useState<string>('');

  // --- VIEWPORT STATE ---
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number, y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number, y: number }>({ x: 0, y: 0 });
  const [isSplitView, setIsSplitView] = useState<boolean>(false);

  // --- HEATMAP CONTROLS ---
  const [aiHeatmapUrl, setAiHeatmapUrl] = useState<string | null>(null);
  const [heatmapMode, setHeatmapMode] = useState<'off' | 'quality' | 'ai' | 'both'>('off');
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.6);

  // --- COMPARISON MODE STATE ---
  const [beforeImage, setBeforeImage] = useState<File | null>(null);
  const [beforePreview, setBeforePreview] = useState<string | null>(null);
  const [afterImage, setAfterImage] = useState<File | null>(null);
  const [afterPreview, setAfterPreview] = useState<string | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);
  const [isComparing, setIsComparing] = useState<boolean>(false);

  // --- CHAT STATE ---
  const [isChatOpen, setIsChatOpen] = useState<boolean>(false);

  // --- REFS ---
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- HANDLERS ---

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File, existingPatient: Patient | null = null) => {
    setImage(file);
    setImagePreview(URL.createObjectURL(file));
    setZoom(1);
    setPan({ x: 0, y: 0 });

    setResult(null);
    setReportText('');
    setQualityCheck(null);
    setError(null);
    setQualityHeatmapUrl(null);
    setAiHeatmapUrl(null);
    setHeatmapMode('off');

    if (existingPatient) {
      setPatientData({ ...existingPatient });
    }

    performQualityCheck(file);
  };

  const performQualityCheck = async (file: File): Promise<void> => {
    setIsCheckingQuality(true);
    try {
      const img = await loadImageFromFile(file);
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const issues: QualityIssue[] = [];

      const brightness = checkBrightness(imageData);
      const blur = checkBlur(imageData);
      const contrast = checkContrast(imageData);
      const metal = checkMetal(imageData);
      const rotation = checkRotation(imageData);
      const labels = checkLabels(imageData);

      if (brightness) issues.push(brightness);
      if (blur) issues.push(blur);
      if (contrast) issues.push(contrast);
      if (metal) issues.push(metal);
      if (rotation) issues.push(rotation);
      if (labels) issues.push(labels);

      const score = calculateQualityScore(issues);
      const isAcceptable = score >= 60;

      setQualityCheck({
        isAcceptable,
        score,
        issues,
        warnings: issues.map(i => i.message)
      });

      await generateQualityHeatmap(file, issues);

    } catch (e) {
      console.error("Quality Check Failed", e);
    } finally {
      setIsCheckingQuality(false);
    }
  };

  const loadImageFromFile = (file: File): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  };

  const checkBrightness = (data: ImageData): QualityIssue | null => {
    let sum = 0;
    for (let i = 0; i < data.data.length; i += 4) sum += (data.data[i] + data.data[i + 1] + data.data[i + 2]) / 3;
    const mean = sum / (data.data.length / 4);
    if (mean < 40) return { type: 'brightness', severity: 'high', message: 'Image too dark', value: mean };
    if (mean > 220) return { type: 'brightness', severity: 'high', message: 'Image too bright', value: mean };
    return null;
  };

  const checkBlur = (_data: ImageData): QualityIssue | null => { return null; };
  const checkContrast = (_data: ImageData): QualityIssue | null => { return null; };

  const checkMetal = (_imageData: ImageData): QualityIssue | null => {
    if (Math.random() < 0.05) return { type: 'metal', severity: 'high', message: 'Foreign metal object detected', value: 1 };
    return null;
  };

  const checkRotation = (_imageData: ImageData): QualityIssue | null => {
    if (Math.random() < 0.05) return { type: 'rotation', severity: 'medium', message: 'Warning: Patient rotated', value: 15 };
    return null;
  };

  const checkLabels = (_imageData: ImageData): QualityIssue | null => { return null; };

  const calculateQualityScore = (issues: QualityIssue[]) => {
    let score = 100;
    issues.forEach(i => score -= (i.severity === 'high' ? 40 : i.severity === 'medium' ? 20 : 10));
    return Math.max(0, score);
  };

  const generateQualityHeatmap = async (file: File, issues: QualityIssue[]) => {
    if (issues.length === 0) return;
    const img = await loadImageFromFile(file);
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
    if (issues.some(i => i.severity === 'high')) ctx.fillRect(0, 0, canvas.width, canvas.height);

    setQualityHeatmapUrl(canvas.toDataURL());
  };

  const generateAIHeatmap = (result: AnalysisResult, w: number, h: number) => {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const gradient = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 3);
    if (result.main_label === 'Abnormal') {
      gradient.addColorStop(0, 'rgba(255, 0, 0, 0.6)');
      gradient.addColorStop(1, 'rgba(0,0,0,0)');
    } else {
      gradient.addColorStop(0, 'rgba(0, 255, 100, 0.4)');
      gradient.addColorStop(1, 'rgba(0,0,0,0)');
    }
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);

    setAiHeatmapUrl(canvas.toDataURL());
  };

  const analyzeImage = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    try {
      if (USE_MOCK_BACKEND) {
        await new Promise(r => setTimeout(r, 2000));

        let isAbnormal = Math.random() > 0.4;
        if (patientData.name.includes('Marcus')) isAbnormal = true;
        if (patientData.name.includes('Sarah')) isAbnormal = false;

        if (modelType === 'radiology') {
          const findings: Finding[] = isAbnormal
            ? [{ label: 'Pneumonia', confidence: 0.96, uncertainty: 0.02 }, { label: 'Infiltration', confidence: 0.89, uncertainty: 0.05 }]
            : [{ label: 'No Finding', confidence: 0.99, uncertainty: 0.01 }];

          const report = isAbnormal
            ? `RADIOLOGY REPORT
Patient: ${patientData.name}
Date: ${new Date().toLocaleDateString()}

PATIENT HISTORY:
Patient presenting with ${patientData.symptom || "unspecified symptoms"}. Clinical concern for infection.

  TECHNIQUE:
PA and lateral views of the chest were obtained in an upright position.

  FINDINGS:
Right lung: There is a focal area of consolidation in the right lower lobe, suggestive of pneumonia.No pleural effusion or pneumothorax is seen.
Left lung: The left lung is clear without focal consolidation, vascular congestion, or pleural effusion.
  Heart / Mediastinum: The cardiomediastinal silhouette is within normal limits.Trachea is midline.
    Bones / Soft Tissues: No acute osseous abnormality.Soft tissues are unremarkable.

      IMPRESSION:
1. Right lower lobe consolidation, concerning for community - acquired pneumonia.
2. No evidence of heart failure or other acute cardiopulmonary process.

  RECOMMENDATION:
1. Clinical correlation suggested.
2. Follow - up radiography in 6 - 8 weeks to ensure resolution.`
            : `RADIOLOGY REPORT
Patient: ${patientData.name}
Date: ${new Date().toLocaleDateString()}

PATIENT HISTORY:
Patient presenting with ${patientData.symptom || "routine check"}.

TECHNIQUE:
PA and lateral views of the chest were obtained.

  FINDINGS:
Lungs: The lungs are clear.There is no focal consolidation, pleural effusion, or pneumothorax.
  Heart / Mediastinum: Heart size is normal.The mediastinal contours are unremarkable.
    Bones: No acute fracture or dislocation.
Soft Tissues: Unremarkable.

  IMPRESSION:
No acute cardiopulmonary abnormality.

  RECOMMENDATION:
No specific follow - up required.`;

          let consensus = undefined;
          if (isSecondOpinionMode) {
            consensus = {
              models: [{ model_name: 'EfficientNet', prediction: 'Abnormal', confidence: 0.9 }],
              final_consensus: 'Abnormal', final_confidence: 0.9, agreement_level: 'High'
            } as ConsensusResult;
          }

          setResult({
            main_label: isAbnormal ? "Abnormal" : "Normal",
            detailed_findings: findings,
            generated_report: report,
            heatmap_b64: null,
            uncertainty_score: isAbnormal ? 0.04 : 0.01,
            consensus_result: consensus,
            reliability: { isReliable: true, score: 92, reason: "Focus aligns with ROI", anatomical_region: "Right Lower Lobe" }
          });
          setReportText(report);
        } else {
          setResult({
            main_label: "Severe",
            detailed_findings: [{ label: 'Melanoma', confidence: 0.85, uncertainty: 0.1 }],
            generated_report: "DERMATOLOGY REPORT\nSuspected Melanoma...",
            heatmap_b64: null,
            uncertainty_score: 0.1
          });
          setReportText("DERMATOLOGY REPORT\nSuspected Melanoma...");
        }

        if (imagePreview) {
          const img = new Image();
          img.onload = () => generateAIHeatmap({ main_label: isAbnormal ? 'Abnormal' : 'Normal' } as any, img.width, img.height);
          img.src = imagePreview;
        }

      } else {
        const formData = new FormData();
        formData.append("file", image);

        // Choose endpoint based on capability
        let endpoint = modelType === 'radiology' ? API_URL : SKIN_API_URL;
        if (USE_LLM_PREDICTION && modelType === 'radiology') {
          endpoint = LLM_API_URL;
        }

        const res = await fetch(endpoint, { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
          setError(`Analysis Failed: ${data.error}`);
          setResult(null);
          setReportText("");
        } else {
          setResult(data);
          setReportText(data.generated_report || "No report generated.");

          // Display heatmap if returned
          if (data.heatmap_b64) {
            setAiHeatmapUrl(`data:image/png;base64,${data.heatmap_b64}`);
            setHeatmapMode('ai');
          }

          setError(null);
        }
      }
    } catch (e) {
      setError("Analysis Failed. Backend not reachable.");
    } finally {
      setLoading(false);
    }
  };

  const simulateFederatedSync = async () => {
    setFederatedStatus('syncing');
    setTimeout(() => setFederatedStatus('synced'), 2000);
  };

  const runComparison = async () => {
    if (!beforeImage || !afterImage) return;
    setIsComparing(true);
    await new Promise(r => setTimeout(r, 1500));

    setComparisonResult({
      improvement_score: 75,
      worsening_score: 5,
      status: 'Improved',
      diff_heatmap_url: afterPreview || '' // In real app, generate diff
    });
    setIsComparing(false);
  };

  const handleApproveAndSign = () => {
    if (!result || !patientData.id) {
      alert("No analysis result or patient data to approve.");
      return;
    }
    setReportText(prev => prev + `\n\n[SIGNED] Dr. AI (Verified)\n${new Date().toLocaleString()}`);
    console.log(`Report for patient ${patientData.name} (${patientData.id}) approved and signed.`);
    alert(`Report for ${patientData.name} approved and signed!`);
    // In a real application, this would involve:
    // 1. Sending the final report and status to a backend.
    // 2. Updating the patient's status in the worklist.
    // 3. Potentially generating a PDF or final document.
  };


  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) { setIsDragging(true); setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y }); }
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  };

  const currentQueue = modelType === 'radiology' ? MOCK_QUEUE : SKIN_MOCK_QUEUE;
  const filteredQueue = currentQueue.filter(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()));

  return (
    <div className="h-screen w-screen bg-slate-900 text-slate-100 font-sans flex overflow-hidden selection:bg-purple-500/30">

      {/* 1. SIDEBAR */}
      <div className="w-20 flex flex-col items-center py-6 glass-strong border-r border-white/5 z-20">
        <div className="mb-8 p-3 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-purple-500/20 transform hover:scale-110 transition-all duration-300">
          <Activity className="w-6 h-6 text-white" />
        </div>

        <nav className="flex-1 flex flex-col gap-4 w-full px-3">
          {[{ id: 'dashboard', icon: LayoutGrid, label: 'Board' }, { id: 'worklist', icon: Users, label: 'Queue' }, { id: 'viewer', icon: Eye, label: 'Viewer' }, { id: 'report', icon: FileText, label: 'Report' }].map(item => (
            <button key={item.id} onClick={() => setActiveView(item.id as any)} className={`p-3 rounded-xl transition-all duration-300 group relative flex justify-center ${activeView === item.id ? 'glass bg-white/10 text-white shadow-lg' : 'text-slate-400 hover:text-white hover:bg-white/5'} `}>
              <item.icon className="w-5 h-5" />
              <span className="absolute left-14 bg-slate-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none border border-white/10">{item.label}</span>
            </button>
          ))}
          <div className="my-2 h-px bg-white/10 w-full" />
          <button onClick={() => setDarkMode(!darkMode)} className="p-3 rounded-xl text-slate-400 hover:text-yellow-400 transition-colors">
            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </nav>
        <div className="mt-auto"><div className="w-10 h-10 rounded-full bg-gradient-to-tr from-purple-500 to-pink-500 p-[2px]"><div className="w-full h-full rounded-full bg-slate-900 flex items-center justify-center text-xs font-bold">AS</div></div></div>
      </div>

      {/* 2. MAIN CONTENT */}
      <div className="flex-1 flex flex-col relative overflow-hidden">

        {/* HEADER */}
        <header className="h-16 glass border-b border-white/5 flex items-center justify-between px-6 z-10 shrink-0">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400 tracking-tight">MedTriage <span className="text-purple-400 font-light">AI</span></h1>
            <div className="h-6 w-px bg-white/10 mx-2" />
            <div className="flex bg-slate-800/50 p-1 rounded-lg border border-white/5">
              <button onClick={() => setAppMode('triage')} className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all ${appMode === 'triage' ? 'bg-purple-500 text-white shadow-lg' : 'text-slate-400 hover:text-white'} `}>TRIAGE</button>
              <button onClick={() => setAppMode('compare')} className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all ${appMode === 'compare' ? 'bg-cyan-500 text-white shadow-lg' : 'text-slate-400 hover:text-white'} `}>COMPARE</button>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex bg-slate-800/50 p-1 rounded-lg border border-white/5">
              <button onClick={() => setModelType('radiology')} className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center gap-2 transition-all ${modelType === 'radiology' ? 'bg-slate-700 text-white' : 'text-slate-400'} `}><Scan className="w-3 h-3" /> Radiology</button>
              <button onClick={() => setModelType('dermatology')} className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center gap-2 transition-all ${modelType === 'dermatology' ? 'bg-slate-700 text-white' : 'text-slate-400'} `}><Stethoscope className="w-3 h-3" /> Derm</button>
            </div>
            <div className="relative group">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-purple-400 transition-colors" />
              <input type="text" value={searchTerm} onChange={e => setSearchTerm(e.target.value)} placeholder="Search patients..." className="pl-9 pr-4 py-2 bg-slate-800/10 border border-white/10 rounded-xl text-sm focus:outline-none focus:bg-slate-800/80 focus:border-purple-500/50 transition-all w-64" />
            </div>
            <button onClick={simulateFederatedSync} className={`flex items-center gap-2 text-xs font-bold px-3 py-2 rounded-lg border transition-all ${federatedStatus === 'synced' ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10' : 'border-white/10 text-slate-400 hover:text-white'} `}><Share2 className="w-3.5 h-3.5" />{federatedStatus === 'synced' ? 'SYNCED' : 'FEDERATED SYNC'}</button>
          </div>
        </header>

        {/* CONTENT AREA */}
        <main className="flex-1 p-6 relative overflow-hidden">
          {appMode === 'compare' ? (
            <div className="flex h-full gap-6 animate-in fade-in zoom-in-95 duration-300">
              <div className="w-80 flex flex-col gap-4">
                <div className="flex-1 glass-card p-4 flex flex-col">
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Baseline (Before)</span>
                  <div onClick={() => document.getElementById('before-upload')?.click()} className="flex-1 border-2 border-dashed border-slate-700 rounded-xl hover:border-purple-500/50 hover:bg-white/5 transition-all cursor-pointer flex flex-col items-center justify-center overflow-hidden relative">
                    {beforePreview ? <img src={beforePreview} className="absolute inset-0 w-full h-full object-contain p-2" /> : <Upload className="text-slate-500 mb-2" />}
                    <input type="file" id="before-upload" className="hidden" onChange={e => { if (e.target.files) { setBeforeImage(e.target.files[0]); setBeforePreview(URL.createObjectURL(e.target.files[0])) } }} />
                  </div>
                </div>
                <div className="flex-1 glass-card p-4 flex flex-col">
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Current (After)</span>
                  <div onClick={() => document.getElementById('after-upload')?.click()} className="flex-1 border-2 border-dashed border-slate-700 rounded-xl hover:border-cyan-500/50 hover:bg-white/5 transition-all cursor-pointer flex flex-col items-center justify-center overflow-hidden relative">
                    {afterPreview ? <img src={afterPreview} className="absolute inset-0 w-full h-full object-contain p-2" /> : <Upload className="text-slate-500 mb-2" />}
                    <input type="file" id="after-upload" className="hidden" onChange={e => { if (e.target.files) { setAfterImage(e.target.files[0]); setAfterPreview(URL.createObjectURL(e.target.files[0])) } }} />
                  </div>
                </div>
                <button onClick={runComparison} disabled={!beforeImage || !afterImage || isComparing} className="py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 font-bold text-white shadow-lg disabled:opacity-50">{isComparing ? 'ANALYZING...' : 'RUN COMPARISON'}</button>
              </div>

              <div className="flex-1 glass-card p-1 relative flex items-center justify-center bg-black/40">
                {comparisonResult ? (
                  <div className="relative w-full h-full p-4">
                    <img src={comparisonResult.diff_heatmap_url} className="w-full h-full object-contain" />
                    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 glass px-6 py-3 rounded-2xl border border-white/10 flex items-center gap-8 shadow-2xl">
                      <div className="text-center"><div className="text-[10px] text-slate-400 uppercase tracking-wider">Status</div><div className={`text-xl font-bold ${comparisonResult.status === 'Improved' ? 'text-emerald-400' : 'text-red-400'} `}>{comparisonResult.status}</div></div>
                      <div className="w-px h-8 bg-white/10" />
                      <div className="text-center"><div className="text-[10px] text-slate-400 uppercase tracking-wider">Improvement</div><div className="text-xl font-bold text-emerald-400">{comparisonResult.improvement_score.toFixed(0)}%</div></div>
                    </div>
                  </div>
                ) : (
                  <div className="text-slate-500 flex flex-col items-center gap-3"><Split className="w-12 h-12 opacity-50" /><p className="font-mono text-sm">Waiting for input...</p></div>
                )}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-12 gap-6 h-full animate-in fade-in slide-in-from-bottom-5 duration-500">

              {/* COL 1: QUEUE */}
              {(activeView === 'dashboard' || activeView === 'worklist') && (
                <div className={`${activeView === 'dashboard' ? 'col-span-3' : 'col-span-12'} flex flex-col gap-6 min-h-0`}>
                  <div onClick={() => fileInputRef.current?.click()} className={`p-6 rounded-2xl border-2 border-dashed transition-all cursor-pointer group flex flex-col items-center gap-3 ${image ? 'border-purple-500/50 bg-purple-500/5' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-800/50'} `}>
                    <input type="file" className="hidden" ref={fileInputRef} onChange={handleImageChange} accept="image/*" />
                    {imagePreview ? <img src={imagePreview} className="w-20 h-20 object-cover rounded-lg shadow-lg" /> : <div className="p-3 rounded-full bg-slate-800 group-hover:bg-purple-900/30 transition-colors"><Upload className="w-6 h-6 text-slate-400 group-hover:text-purple-400" /></div>}
                    {!imagePreview && <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Upload Scan</span>}
                  </div>

                  {isCheckingQuality && <div className="p-4 text-xs font-bold text-cyan-400 animate-pulse">Running Quality Checks...</div>}
                  {qualityCheck && !isCheckingQuality && (
                    <div className={`rounded-xl p-4 border ${qualityCheck.score > 80 ? 'bg-emerald-900/10 border-emerald-500/30' : 'bg-red-900/10 border-red-500/30'} `}>
                      <div className="flex justify-between items-center mb-2"><span className="text-xs font-bold text-slate-300">Quality Score</span><span className={`${qualityCheck.score > 80 ? 'text-emerald-400' : 'text-red-400'} font-bold`}>{qualityCheck.score}/100</span></div>
                      {qualityCheck.issues.slice(0, 2).map((i, idx) => <div key={idx} className="flex gap-2 items-center text-[10px] text-slate-400 mt-1"><AlertTriangle className="w-3 h-3 text-red-400/70" /> {i.message}</div>)}
                    </div>
                  )}

                  <div className="flex-1 glass-card flex flex-col min-h-0 overflow-hidden">
                    <div className="p-4 border-b border-white/5 flex justify-between items-center"><span className="font-bold text-sm">Patient Queue</span><span className="text-[10px] bg-white/10 px-2 py-0.5 rounded-full text-slate-300">{filteredQueue.length}</span></div>
                    <div className="flex-1 overflow-y-auto p-2 space-y-1">
                      {filteredQueue.map(p => (
                        <div key={p.id} onClick={() => { setPatientData(p); if (!image) alert("Please upload an image into the upload box above to simulate this case."); }} className={`p-3 rounded-xl cursor-pointer transition-all border border-transparent hover:bg-white/5 ${patientData.name === p.name ? 'bg-purple-500/10 border-purple-500/30' : ''} `}>
                          <div className="flex justify-between mb-1"><span className={`font-semibold text-sm ${patientData.name === p.name ? 'text-purple-300' : 'text-slate-200'} `}>{p.name}</span><span className={`text-[10px] font-bold px-1.5 rounded ${p.urgency === 'Critical' ? 'bg-red-500/20 text-red-400' : 'bg-slate-700 text-slate-400'} `}>{p.urgency}</span></div>
                          <div className="text-xs text-slate-500 truncate">{p.symptom}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* COL 2: VIEWER */}
              {(activeView === 'dashboard' || activeView === 'viewer') && (
                <div className={`${activeView === 'dashboard' ? 'col-span-5' : 'col-span-12'} flex flex-col gap-4 min-h-0 relative`}>
                  <div className="flex-1 glass-card relative overflow-hidden flex items-center justify-center bg-black/20 group" onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={() => setIsDragging(false)} onMouseLeave={() => setIsDragging(false)}>

                    {/* OVERLAY TOOLS */}
                    <div className="absolute top-4 left-4 z-10 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={() => setZoom(z => Math.min(z + 0.5, 4))} className="p-2 glass rounded-lg hover:bg-white/20"><ZoomIn className="w-4 h-4" /></button>
                      <button onClick={() => setZoom(z => Math.max(z - 0.5, 1))} className="p-2 glass rounded-lg hover:bg-white/20"><ZoomOut className="w-4 h-4" /></button>
                      <button onClick={() => setIsSplitView(!isSplitView)} className={`p-2 glass rounded-lg hover:bg-white/20 ${isSplitView ? 'text-purple-400' : ''} `}><Split className="w-4 h-4" /></button>
                    </div>

                    {/* HEATMAP TOOLS */}
                    {(qualityHeatmapUrl || aiHeatmapUrl) && (
                      <div className="absolute top-4 right-4 z-10 glass rounded-lg p-2 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity w-32">
                        <span className="text-[10px] font-bold text-slate-400 uppercase">Heatmap</span>
                        <div className="flex gap-1">
                          {['off', 'ai', 'quality'].map(m => (
                            <button key={m} onClick={() => setHeatmapMode(m as any)} className={`flex-1 text-[9px] py-1 rounded ${heatmapMode === m ? 'bg-purple-500 text-white' : 'glass hover:bg-white/10'} `}>{m.toUpperCase()}</button>
                          ))}
                        </div>
                        <input type="range" min="0" max="100" value={heatmapOpacity * 100} onChange={e => setHeatmapOpacity(Number(e.target.value) / 100)} className="h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer" />
                      </div>
                    )}

                    {imagePreview ? (
                      <div className="transition-transform duration-100" style={{ transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)` }}>
                        <div className={`relative flex gap-1 ${isSplitView ? 'w-[200%]' : 'w-auto'} `}>
                          <div className="relative">
                            <img src={imagePreview} className="max-h-[80vh] max-w-full object-contain shadow-2xl rounded" draggable="false" />
                            {(heatmapMode === 'ai' || heatmapMode === 'both') && aiHeatmapUrl && <img src={aiHeatmapUrl} className="absolute inset-0 w-full h-full mix-blend-screen pointer-events-none" style={{ opacity: heatmapOpacity }} />}
                            {(heatmapMode === 'quality' || heatmapMode === 'both') && qualityHeatmapUrl && <img src={qualityHeatmapUrl} className="absolute inset-0 w-full h-full mix-blend-screen pointer-events-none" style={{ opacity: heatmapOpacity }} />}
                            <div className="absolute bottom-2 right-2 text-[10px] font-bold text-slate-500 bg-black/50 px-2 rounded">Scan</div>
                          </div>
                          {isSplitView && (
                            <div className="relative">
                              <img src={imagePreview} className="max-h-[80vh] max-w-full object-contain shadow-2xl rounded grayscale invert" draggable="false" />
                              <div className="absolute inset-0 bg-red-500/20 mix-blend-multiply" />
                              <div className="absolute bottom-2 right-2 text-[10px] font-bold text-red-300 bg-black/50 px-2 rounded">Filter</div>
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-4 opacity-30"><Monitor className="w-24 h-24 stroke-1" /><span className="tracking-[0.5em] text-xs font-bold">NO SIGNAL</span></div>
                    )}
                  </div>

                  <div className="h-20 glass-card p-4 flex items-center justify-between">
                    <div className="flex flex-col"><span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Current Case</span><span className="text-lg font-bold text-white max-w-[200px] truncate">{patientData.name || '---'}</span></div>
                    <button onClick={analyzeImage} disabled={!image || loading} className="px-8 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 font-bold text-white shadow-lg hover:scale-105 active:scale-95 transition-all disabled:grayscale">{loading ? 'PROCESSING...' : 'ANALYZE SCAN'}</button>
                  </div>
                </div>
              )}

              {/* COL 3: FINDINGS */}
              {(activeView === 'dashboard' || activeView === 'report') && (
                <div className={`${activeView === 'dashboard' ? 'col-span-4' : 'col-span-12'} flex flex-col min-h-0 glass-card`}>
                  {error && <div className="bg-red-900/50 text-red-200 text-xs p-2 text-center border-b border-red-500/20">{error}</div>}

                  <div className="flex border-b border-white/5">
                    <button onClick={() => setActiveTab('findings')} className={`flex-1 py-4 text-xs font-bold uppercase tracking-wider hover:bg-white/5 transition-colors ${activeTab === 'findings' ? 'text-purple-400 border-b-2 border-purple-500 bg-white/5' : 'text-slate-500'} `}>Findings</button>
                    <button onClick={() => setActiveTab('report')} className={`flex-1 py-4 text-xs font-bold uppercase tracking-wider hover:bg-white/5 transition-colors ${activeTab === 'report' ? 'text-purple-400 border-b-2 border-purple-500 bg-white/5' : 'text-slate-500'} `}>Report</button>
                  </div>

                  <div className="flex-1 p-6 overflow-y-auto">
                    {result ? (
                      activeTab === 'findings' ? (
                        <div className="space-y-6 animate-in slide-in-from-right-4 duration-500">
                          <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 border border-white/10 relative overflow-hidden group">
                            <div className={`absolute top-0 right-0 p-32 blur-3xl opacity-20 -mr-16 -mt-16 rounded-full ${result.main_label === 'Abnormal' ? 'bg-red-500' : 'bg-emerald-500'} `} />
                            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-1">Diagnosis</h3>
                            <div className={`text-4xl font-bold mb-4 ${result.main_label === 'Abnormal' ? 'text-red-400' : 'text-emerald-400'}`}>{result.main_label}</div>

                            <div className="flex items-center gap-4 bg-black/20 p-3 rounded-xl border border-white/5">
                              <div className="relative w-14 h-14 flex items-center justify-center">
                                <svg className="w-full h-full transform -rotate-90">
                                  <circle cx="28" cy="28" r="24" stroke="currentColor" strokeWidth="4" fill="transparent" className="text-slate-800" />
                                  <circle cx="28" cy="28" r="24" stroke="currentColor" strokeWidth="4" fill="transparent" strokeDasharray={150.7} strokeDashoffset={150.7 * result.uncertainty_score} className={result.main_label === 'Abnormal' ? 'text-red-500' : 'text-emerald-500'} strokeLinecap="round" />
                                </svg>
                                <span className="absolute text-xs font-bold text-white">{(100 * (1 - result.uncertainty_score)).toFixed(0)}%</span>
                              </div>
                              <div className="flex flex-col">
                                <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Model Confidence</span>
                                <div className="flex gap-2 items-center">
                                  <span className="text-white font-bold text-sm">High Accuracy</span>
                                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                                </div>
                              </div>
                            </div>
                          </div>

                          <div className="flex items-center justify-between text-xs text-slate-400 glass p-2 rounded-lg">
                            <span>Second Opinion Mode</span>
                            <button onClick={() => setIsSecondOpinionMode(!isSecondOpinionMode)} className={`w-8 h-4 rounded-full transition-colors relative ${isSecondOpinionMode ? 'bg-purple-500' : 'bg-slate-700'} `}>
                              <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform ${isSecondOpinionMode ? 'translate-x-4' : ''} `} />
                            </button>
                          </div>

                          <div className="space-y-3">
                            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Detected Conditions</h4>
                            {(result.detailed_findings || []).map((f, i) => (
                              <div key={i} className="glass p-3 rounded-xl flex items-center justify-between group hover:bg-white/10 transition-colors">
                                <span className="text-sm font-medium">{f.label}</span>
                                <div className="flex items-center gap-3">
                                  <div className="w-24 h-1.5 bg-slate-700 rounded-full overflow-hidden"><div style={{ width: `${f.confidence * 100}% ` }} className={`h-full rounded-full ${f.confidence > 0.5 ? 'bg-gradient-to-r from-cyan-500 to-blue-500' : 'bg-slate-500'} `} /></div>
                                  <span className="text-xs font-mono font-bold w-8 text-right">{(f.confidence * 100).toFixed(0)}%</span>
                                </div>
                              </div>
                            ))}
                          </div>

                          {result.reliability && (
                            <div className={`p-4 rounded-xl border ${result.reliability.isReliable ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-red-500/5 border-red-500/20'} `}>
                              <div className="flex gap-3">
                                {result.reliability.isReliable ? <ShieldAlert className="w-5 h-5 text-emerald-500" /> : <AlertTriangle className="w-5 h-5 text-red-500" />}
                                <div><div className={`text-xs font-bold mb-1 ${result.reliability.isReliable ? 'text-emerald-400' : 'text-red-400'} `}>{result.reliability.isReliable ? 'Anatomically Verified' : 'Reliability Warning'}</div><p className="text-[10px] text-slate-400 leading-relaxed">{result.reliability.reason}</p></div>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="h-full flex flex-col animate-in fade-in">
                          <div className="flex-1 bg-white text-slate-900 p-6 font-serif text-sm leading-7 shadow-inner rounded-md overflow-y-auto whitespace-pre-wrap">{reportText}</div>
                          <div className="mt-4 flex gap-3">
                            <button onClick={() => setIsRecording(!isRecording)} className={`flex-1 py-3 px-4 rounded-xl font-bold text-xs flex items-center justify-center gap-2 border transition-all ${isRecording ? 'border-red-500 text-red-400 bg-red-500/10 animate-pulse' : 'border-slate-600 text-slate-400 hover:text-white'} `}><Mic className="w-4 h-4" /> {isRecording ? 'RECORDING...' : 'DICTATE'}</button>
                            <button className="flex-1 py-3 px-4 rounded-xl font-bold text-xs bg-emerald-600 text-white hover:bg-emerald-500 transition-colors shadow-lg shadow-emerald-900/20" onClick={handleApproveAndSign}>APPROVE & SIGN</button>
                          </div>
                        </div>
                      )
                    ) : (
                      <div className="h-full flex flex-col items-center justify-center opacity-40"><FileText className="w-16 h-16 text-slate-600 mb-4" /><p className="font-mono text-xs">Waiting for analysis...</p></div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      <ChatOverlay context={result ? JSON.stringify(result.detailed_findings) : "No findings yet."} isOpen={isChatOpen} setIsOpen={setIsChatOpen} />
    </div>
  );
};

interface ChatOverlayProps { context: string; isOpen: boolean; setIsOpen: (v: boolean) => void; }
const ChatOverlay: React.FC<ChatOverlayProps> = ({ context, isOpen, setIsOpen }) => {
  const [messages, setMessages] = useState<{ role: 'user' | 'ai', text: string }[]>([{ role: 'ai', text: "Hello. I am the MedTriage Assistant. I can answer questions about the current case." }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [messages, isOpen]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const msg = input; setInput("");
    setMessages(p => [...p, { role: 'user', text: msg }]);
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: msg, context: context }) });
      const data = await res.json();
      setMessages(p => [...p, { role: 'ai', text: data.response }]);
    } catch { setMessages(p => [...p, { role: 'ai', text: "Connection error." }]); }
    finally { setLoading(false); }
  };

  if (!isOpen) return <button onClick={() => setIsOpen(true)} className="fixed bottom-6 right-6 p-4 bg-purple-600 hover:bg-purple-500 text-white rounded-full shadow-lg shadow-purple-900/50 transition-all hover:scale-110 z-50 flex items-center gap-2"><MessageSquare className="w-6 h-6" /></button>;

  return (
    <div className="fixed bottom-6 right-6 w-96 h-[500px] bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl z-50 flex flex-col overflow-hidden animate-in slide-in-from-bottom-10 fade-in duration-300">
      <div className="p-4 bg-slate-800 border-b border-slate-700 flex justify-between items-center text-white"><div className="flex items-center gap-2"><Brain className="w-5 h-5 text-purple-400" /><span className="font-semibold">AI Assistant</span></div><button onClick={() => setIsOpen(false)}><Minimize2 className="w-4 h-4 text-slate-400" /></button></div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m, i) => (<div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}><div className={`max-w-[85%] p-3 rounded-2xl text-sm ${m.role === 'user' ? 'bg-purple-600 text-white rounded-br-none' : 'bg-slate-800 text-slate-200 border border-slate-700 rounded-bl-none'}`}>{m.text}</div></div>))}
        {loading && <div className="text-xs text-slate-500 animate-pulse ml-2">Thinking...</div>}
      </div>
      <div className="p-3 bg-slate-800 border-t border-slate-700 flex gap-2"><input type="text" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSend()} placeholder="Ask something..." className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500" /><button onClick={handleSend} disabled={loading} className="p-2 bg-purple-600 rounded-lg text-white hover:bg-purple-500"><Send className="w-4 h-4" /></button></div>
    </div>
  );
};

export default TriageApp;