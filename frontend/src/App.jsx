import React, { useState, useEffect } from 'react';

function App() {
  const [mode, setMode] = useState("manual"); 
  const [status, setStatus] = useState("System Ready");
  
  // Settings
  const [gridSize, setGridSize] = useState(10);
  const [hazardCount, setHazardCount] = useState(15);
  const [batchCount, setBatchCount] = useState(100);
  
  // Manual Mode States
  const [grid, setGrid] = useState(Array(100).fill(null));
  const [aiData, setAiData] = useState({ 
    heatmap: Array(100).fill(0), 
    cnnConfidence: 0, 
    bayesianEst: 0,
    safestTileIndex: -1 
  });

  // Batch Mode States
  const [batchResults, setBatchResults] = useState(null);
  const [winPage, setWinPage] = useState(0);
  const [lossPage, setLossPage] = useState(0);
  
  // XAI States
  const [selectedSim, setSelectedSim] = useState(null);
  const [geminiExplanation, setGeminiExplanation] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  // --- MANUAL MODE LOGIC ---
  const startMission = async () => {
    try {
      setStatus("Initializing...");
      const resp = await fetch(`http://127.0.0.1:8000/initialize-mission?size=${gridSize}&hazards=${hazardCount}`);
      if (resp.ok) {
        setGrid(Array(gridSize * gridSize).fill(null)); 
        setAiData({ heatmap: Array(gridSize * gridSize).fill(0), cnnConfidence: 0, bayesianEst: 0, safestTileIndex: -1 });
        setStatus("System Ready");
      }
    } catch (error) {
      setStatus("OFFLINE: START BACKEND");
    }
  };

  const getAIRiskAnalysis = async (currentGrid) => {
    try {
      const resp = await fetch("http://127.0.0.1:8000/analyze-risk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ grid: currentGrid }),
      });
      const data = await resp.json();
      setAiData({ 
        heatmap: data.heatmap, 
        cnnConfidence: data.cnn_confidence, 
        bayesianEst: data.bayesian_est,
        safestTileIndex: data.safest_tile_index
      });
    } catch (error) { console.error("AI Analysis Failed", error); }
  };

  const handleTileClick = async (index) => {
    if (status.includes("FAILURE") || status.includes("CLEARED") || grid[index] !== null) return;
    
    const r = Math.floor(index / gridSize);
    const c = index % gridSize;
    
    try {
      const resp = await fetch(`http://127.0.0.1:8000/scan-tile?r=${r}&c=${c}`);
      const data = await resp.json();
      
      const newGrid = [...grid];
      
      if (data.is_hazard) {
        // Calculate clearance %
        const cleared = newGrid.filter(x => typeof x === 'number').length;
        const totalSafe = (gridSize * gridSize) - hazardCount;
        const percent = ((cleared / totalSafe) * 100).toFixed(1);
        
        // Reveal everything
        const revealedField = data.full_field.map((val, i) => {
            if (i === index) return "X"; // The one you hit
            if (val === -1) return "H"; // Other hidden hazards
            return val; // Numbers
        });
        
        setGrid(revealedField);
        setStatus(`CRITICAL FAILURE: ${percent}% CLEARED`);
      } else {
        newGrid[index] = data.sensor_reading;
        setGrid(newGrid);
        
        // Check for win
        const cleared = newGrid.filter(x => typeof x === 'number').length;
        if (cleared === (gridSize * gridSize) - hazardCount) {
            setStatus("MISSION SUCCESS: 100% CLEARED");
        } else {
            getAIRiskAnalysis(newGrid);
        }
      }
    } catch (error) { setStatus("SENSOR ERROR"); }
  };

  // --- BATCH MODE LOGIC ---
  const runBatchSolve = async () => {
    try {
      setStatus(`Simulating ${batchCount} missions...`);
      setBatchResults(null);
      setWinPage(0);
      setLossPage(0);
      
      const resp = await fetch("http://127.0.0.1:8000/batch-solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ size: gridSize, hazards: hazardCount, num_games: batchCount }),
      });
      const data = await resp.json();
      setBatchResults(data);
      setStatus("Batch Evaluation Complete");
    } catch (error) { setStatus("BATCH FAILED"); }
  };

  const renderMiniGrid = (simData, idx) => (
    <div 
      key={idx} 
      onClick={async () => {
        setSelectedSim(simData);
        setIsGenerating(true);
        setGeminiExplanation("Establishing uplink...");
        try {
          const resp = await fetch("http://127.0.0.1:8000/generate-explanation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              cleared_percent: simData.telemetry.cleared_percent,
              engine: simData.telemetry.engine,
              raw_explanation: simData.telemetry.explanation
            }),
          });
          const data = await resp.json();
          setGeminiExplanation(data.rich_explanation);
        } catch (error) { setGeminiExplanation("XAI Core offline."); }
        finally { setIsGenerating(false); }
      }}
      className="bg-slate-900 border border-cyan-900 p-2 rounded-sm cursor-pointer hover:border-cyan-400 transition-all"
    >
      <div className="grid gap-[1px] w-full aspect-square" style={{ gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))` }}>
        {simData.board.map((cell, i) => (
          <div key={i} className={`flex items-center justify-center text-[8px] font-bold
            ${cell === -1 ? "bg-slate-800" : cell === -2 ? "bg-cyan-700 text-slate-900" : cell === "X" ? "bg-red-600 text-white" : "bg-slate-700 text-cyan-400"}
          `}>{cell === -2 ? "F" : cell === -1 ? "" : cell}</div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-950 text-cyan-400 font-mono p-4 sm:p-8 flex flex-col items-center">
      {/* Header */}
      <div className="w-full max-w-6xl border-b border-cyan-900 pb-4 mb-8 flex flex-col sm:flex-row justify-between items-end gap-4">
        <div>
          <h1 className="text-4xl font-black tracking-tighter uppercase">MineSearcher v1.0</h1>
          <div className="flex gap-4 mt-2">
            <button onClick={() => setMode("manual")} className={`text-xs tracking-widest px-3 py-1 border transition-colors ${mode === 'manual' ? 'bg-cyan-900 border-cyan-400 text-white' : 'border-cyan-900 text-cyan-700'}`}>MANUAL SCAN</button>
            <button onClick={() => setMode("batch")} className={`text-xs tracking-widest px-3 py-1 border transition-colors ${mode === 'batch' ? 'bg-purple-900 border-purple-400 text-white' : 'border-cyan-900 text-cyan-700'}`}>BATCH EVALUATION</button>
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs uppercase text-slate-500">Telemetry Status</p>
          <p className={`text-sm ${status.includes("SUCCESS") || status.includes("Ready") ? "text-green-500" : "text-red-500"}`}>{status}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 w-full max-w-6xl">
        <div className="lg:col-span-2">
          {mode === "manual" ? (
            <div className="aspect-square bg-slate-900 border-2 border-cyan-900 rounded-sm p-4 relative shadow-2xl">
              <div className="grid gap-1 h-full w-full" style={{ gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))` }}>
                {grid.map((cell, i) => {
                  const risk = aiData.heatmap ? aiData.heatmap[i] : 0;
                  const isSafest = aiData.safestTileIndex === i;
                  return (
                    <div 
                      key={i} onClick={() => handleTileClick(i)}
                      style={{ 
                        backgroundColor: cell === null && risk > 0.2 ? `rgba(239, 68, 68, ${risk * 0.6})` : '',
                        border: isSafest ? '2px solid #22c55e' : ''
                      }}
                      className={`border border-cyan-950 flex items-center justify-center text-sm transition-all cursor-crosshair
                        ${cell === null ? (isSafest ? "bg-green-950/30 animate-pulse" : "hover:bg-cyan-900/40") : "bg-slate-800 text-cyan-400"}
                        ${cell === "X" ? "bg-red-600 text-white animate-pulse" : ""}
                        ${cell === "H" ? "bg-red-950/50 text-red-500 opacity-60" : ""}
                      `}
                    >
                      {cell === "H" ? "!" : (cell !== null ? cell : "")}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="bg-slate-900 border-2 border-cyan-900 p-6 rounded-sm min-h-[500px]">
              {!batchResults ? (
                <div className="h-full flex items-center justify-center text-cyan-800 animate-pulse uppercase tracking-widest font-bold">Awaiting Batch Execution...</div>
              ) : (
                <div className="space-y-8">
                  <div>
                    <div className="flex justify-between items-center mb-4 border-b border-cyan-900 pb-2">
                      <h3 className="text-green-500 font-bold uppercase tracking-widest">Successful Clearances ({batchResults.wins.length})</h3>
                      <div className="flex gap-2">
                        <button disabled={winPage === 0} onClick={() => setWinPage(p => p - 1)} className="px-3 bg-slate-800">&lt;</button>
                        <span className="text-xs self-center">Page {winPage + 1}</span>
                        <button disabled={(winPage + 1) * 5 >= batchResults.wins.length} onClick={() => setWinPage(p => p + 1)} className="px-3 bg-slate-800">&gt;</button>
                      </div>
                    </div>
                    <div className="grid grid-cols-5 gap-3">{batchResults.wins.slice(winPage * 5, (winPage + 1) * 5).map((sim, i) => renderMiniGrid(sim, i))}</div>
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-4 border-b border-cyan-900 pb-2">
                      <h3 className="text-red-500 font-bold uppercase tracking-widest">Critical Failures ({batchResults.losses.length})</h3>
                      <div className="flex gap-2">
                        <button disabled={lossPage === 0} onClick={() => setLossPage(p => p - 1)} className="px-3 bg-slate-800">&lt;</button>
                        <span className="text-xs self-center">Page {lossPage + 1}</span>
                        <button disabled={(lossPage + 1) * 5 >= batchResults.losses.length} onClick={() => setLossPage(p => p + 1)} className="px-3 bg-slate-800">&gt;</button>
                      </div>
                    </div>
                    <div className="grid grid-cols-5 gap-3">{batchResults.losses.slice(lossPage * 5, (lossPage + 1) * 5).map((sim, i) => renderMiniGrid(sim, i))}</div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-6">
          <div className="bg-slate-900 border border-cyan-900 p-6 rounded-sm shadow-xl">
            <h2 className="text-xs uppercase mb-4 text-cyan-700 font-bold tracking-widest">Environment Variables</h2>
            <div className="mb-6 space-y-4">
              <div>
                <label className="text-[10px] text-cyan-500 uppercase flex justify-between"><span>Grid Size</span><span>{gridSize}x{gridSize}</span></label>
                <input type="range" min="5" max="20" value={gridSize} onChange={(e) => setGridSize(Number(e.target.value))} className="w-full accent-cyan-500" />
              </div>
              <div>
                <label className="text-[10px] text-cyan-500 uppercase flex justify-between"><span>Hazards</span><span>{hazardCount}</span></label>
                <input type="range" min="1" max={Math.floor((gridSize * gridSize) * 0.4)} value={hazardCount} onChange={(e) => setHazardCount(Number(e.target.value))} className="w-full accent-cyan-500" />
              </div>
              {mode === "batch" && (
                <div>
                  <label className="text-[10px] text-purple-500 uppercase flex justify-between"><span>Missions</span><span>{batchCount}</span></label>
                  <input type="range" min="10" max="500" step="10" value={batchCount} onChange={(e) => setBatchCount(Number(e.target.value))} className="w-full accent-purple-500" />
                </div>
              )}
            </div>
            <button onClick={mode === "manual" ? startMission : runBatchSolve} className={`w-full py-3 font-bold tracking-widest transition-all ${mode === 'manual' ? 'bg-cyan-950 border border-cyan-500 text-cyan-400' : 'bg-purple-950 border border-purple-500 text-purple-400'}`}>
                {mode === "manual" ? "INITIALIZE SCAN" : "EXECUTE BATCH AI"}
            </button>
          </div>

          <div className="bg-slate-900 border border-cyan-900 p-6 rounded-sm flex-grow shadow-xl">
            <h2 className="text-xs uppercase mb-4 text-cyan-700 font-bold tracking-widest">AI Performance</h2>
            <div className="space-y-6">
              {mode === "manual" && (
                <>
                  <div>
                    <p className="text-[10px] text-slate-500 uppercase mb-2">Aggregated Risk Factor</p>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-red-500 transition-all duration-500" style={{ width: `${aiData.bayesianEst}%` }}></div>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="text-[10px] text-slate-500 uppercase mb-1">CNN Confidence</p>
                      <p className="text-3xl font-light text-cyan-100">{aiData.cnnConfidence}%</p>
                    </div>
                    {aiData.safestTileIndex !== -1 && (
                      <div className="text-right">
                        <p className="text-[10px] text-green-500 uppercase mb-1">Optimal Path</p>
                        <p className="text-xs text-green-400 font-bold">READY</p>
                      </div>
                    )}
                  </div>
                </>
              )}
              {mode === "batch" && (
                <div className="flex flex-col justify-center h-full">
                  <p className="text-[10px] text-slate-500 uppercase mb-2">Batch Win Rate</p>
                  <p className={`text-5xl font-black ${batchResults ? (batchResults.win_rate > 60 ? 'text-green-500' : 'text-yellow-500') : 'text-slate-700'}`}>
                    {batchResults ? `${batchResults.win_rate}%` : "--%"}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* XAI MODAL */}
      {selectedSim && (
        <div className="fixed inset-0 bg-slate-950/90 z-50 flex items-center justify-center p-4 backdrop-blur-sm">
          <div className="bg-slate-900 border border-cyan-500 max-w-4xl w-full p-8 shadow-2xl relative">
            <button onClick={() => setSelectedSim(null)} className="absolute top-4 right-6 text-cyan-700 hover:text-white text-2xl font-bold transition-colors">✕</button>
            <h2 className="text-2xl font-black text-white mb-8 uppercase border-b border-cyan-900 pb-4 tracking-widest">Mission Post-Mortem</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
              <div className="grid gap-1 w-full aspect-square bg-slate-950 border-2 border-cyan-900 p-3" style={{ gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))` }}>
                {selectedSim.board.map((cell, i) => (
                  <div key={i} className={`flex items-center justify-center text-sm font-bold ${cell === -1 ? "bg-slate-800" : cell === -2 ? "bg-cyan-700 text-slate-900" : cell === "X" ? "bg-red-600 text-white animate-pulse" : "bg-slate-700 text-cyan-400"}`}>{cell === -2 ? "F" : cell === -1 ? "" : cell}</div>
                ))}
              </div>
              <div className="flex flex-col gap-8 justify-center">
                <div>
                  <p className="text-[10px] text-slate-400 uppercase mb-2 tracking-widest">Clearance Progress</p>
                  <p className={`text-5xl font-light ${selectedSim.telemetry.cleared_percent === 100 ? "text-green-500" : "text-red-500"}`}>{selectedSim.telemetry.cleared_percent}%</p>
                </div>
                <div>
                  <p className="text-[10px] text-slate-400 uppercase mb-2 tracking-widest">System Reasoning (Gemma 2 27B)</p>
                  <div className={`bg-slate-950 border-l-4 p-4 ${isGenerating ? 'border-purple-500' : 'border-cyan-500'}`}>
                    <p className={`text-sm italic ${isGenerating ? 'text-purple-300 animate-pulse' : 'text-cyan-300'}`}>"{geminiExplanation}"</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;