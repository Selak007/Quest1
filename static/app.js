document.addEventListener("DOMContentLoaded", () => {
    const locateForm = document.getElementById("locateForm");
    const submitBtn = document.getElementById("submitBtn");
    
    const processingPanel = document.getElementById("processingPanel");
    const taskStatusBadge = document.getElementById("taskStatusBadge");
    const progressBar = document.getElementById("progressBar");
    const progressLabel = document.getElementById("progressLabel");
    const progressPercent = document.getElementById("progressPercent");
    const terminalBody = document.getElementById("terminalBody");
    
    const resultsPanel = document.getElementById("resultsPanel");
    const resultFrame = document.getElementById("resultFrame");
    const downloadFrameBtn = document.getElementById("downloadFrameBtn");
    const resultConfidence = document.getElementById("resultConfidence");
    const resultConfidenceStatus = document.getElementById("resultConfidenceStatus");
    const resultTimestamp = document.getElementById("resultTimestamp");
    const resultTimestampSec = document.getElementById("resultTimestampSec");
    const resultFrameNumber = document.getElementById("resultFrameNumber");
    const resultTargetText = document.getElementById("resultTargetText");
    const resultAsrText = document.getElementById("resultAsrText");
    const resultFrameWindow = document.getElementById("resultFrameWindow");
    
    const weightText = document.getElementById("weightText");
    const weightAsr = document.getElementById("weightAsr");
    const weightVad = document.getElementById("weightVad");
    const valText = document.getElementById("valText");
    const valAsr = document.getElementById("valAsr");
    const valVad = document.getElementById("valVad");
    
    const historyList = document.getElementById("historyList");
    
    // Step Elements Checklist
    const steps = {
        download: document.getElementById("step-download"),
        audio: document.getElementById("step-audio"),
        vad: document.getElementById("step-vad"),
        scan: document.getElementById("step-scan"),
        refine: document.getElementById("step-refine"),
        align: document.getElementById("step-align"),
        extract: document.getElementById("step-extract")
    };
    
    let logEventSource = null;
    let activeTaskId = null;
    let currentActiveStep = null;

    // Load history list on startup
    loadHistory();

    // ── Form Submit Handler ──────────────────────────────────────────────────
    locateForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const url = document.getElementById("url").value.trim();
        const target = document.getElementById("target").value.trim();
        
        if (!url || !target) return;
        
        // Disable form and show loading panels
        setFormDisabled(true);
        resultsPanel.classList.add("hidden");
        processingPanel.classList.remove("hidden");
        
        // Reset checklist, progress bar & terminal
        resetSteps();
        updateProgressBar(0, "Submitting task to server...");
        terminalBody.innerHTML = '<div class="log-line system">Submitting localization request...</div>';
        
        try {
            const res = await fetch("/api/locate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url, target })
            });
            
            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || "Server error occurred");
            }
            
            const task = await res.json();
            activeTaskId = task.task_id;
            
            // Start listening to the SSE log stream
            startLogStream(task.task_id);
            loadHistory(); // Refresh history sidebar list
            
        } catch (err) {
            appendLogLine(`[Server] Submission failed: ${err.message}`, "error");
            taskStatusBadge.className = "status-badge failed";
            taskStatusBadge.textContent = "Failed";
            setFormDisabled(false);
        }
    });

    // ── Log Stream (Server-Sent Events) Handler ──────────────────────────────
    function startLogStream(taskId) {
        if (logEventSource) {
            logEventSource.close();
        }
        
        taskStatusBadge.className = "status-badge running";
        taskStatusBadge.textContent = "Running";
        resetSteps();
        updateProgressBar(0, "Job started. Performing initial steps...");
        
        logEventSource = new EventSource(`/api/tasks/${taskId}/logs`);
        
        logEventSource.onmessage = (e) => {
            const line = e.data;
            
            if (line === "[DONE]") {
                logEventSource.close();
                fetchTaskDetails(taskId); // Retrieve complete result
                return;
            }
            
            // Log Line Styling/Parsing
            let styleClass = "log-line";
            if (line.includes("[SCAN]")) {
                styleClass = "log-line scan";
            } else if (line.includes("[REFINE]")) {
                styleClass = "log-line refine";
            } else if (line.includes("Error:") || line.includes("failed")) {
                styleClass = "log-line error";
                if (currentActiveStep) {
                    setStepFailed(currentActiveStep);
                }
            } else if (line.includes("complete") || line.includes("Success") || line.includes("Result")) {
                styleClass = "log-line success";
            } else if (line.includes("WARNING") || line.includes("Warning")) {
                styleClass = "log-line warn";
            } else if (line.startsWith("[Server]")) {
                styleClass = "log-line system";
            }
            
            // Trigger checklist updates based on pipeline stages
            if (line.includes("Stage: Video Download")) {
                setStepActive("download");
            } else if (line.includes("Stage: Audio Extraction")) {
                setStepCompleted("download");
                setStepActive("audio");
            } else if (line.includes("Stage: VAD")) {
                setStepCompleted("audio");
                setStepActive("vad");
            } else if (line.includes("Stage: ASR Phase 1: SCAN")) {
                setStepCompleted("vad");
                setStepActive("scan");
            } else if (line.includes("Stage: ASR Phase 2: REFINE")) {
                setStepCompleted("scan");
                setStepActive("refine");
            } else if (line.includes("Stage: Forced Alignment")) {
                setStepCompleted("refine");
                setStepActive("align");
            } else if (line.includes("Stage: Frame PTS Read") || line.includes("Stage: Fuzzy Phrase Match (refine)")) {
                setStepCompleted("align");
                setStepActive("extract");
            } else if (line.includes("Pipeline complete")) {
                setStepCompleted("extract");
            }

            // Capture time durations from completed stages
            if (line.includes("done in") && currentActiveStep) {
                try {
                    const duration = line.split("done in")[1].trim().replace("?", "").replace("…", "");
                    setStepCompleted(currentActiveStep, duration);
                } catch (err) {
                    setStepCompleted(currentActiveStep);
                }
            }

            // Update percentage indicators for active scans
            if (line.includes("% done") || line.includes("% complete") || line.includes("% done")) {
                try {
                    const parts = line.split("%");
                    const left = parts[0];
                    const startIdx = Math.max(left.lastIndexOf(","), left.lastIndexOf("(")) + 1;
                    const pctVal = parseFloat(left.substring(startIdx).trim());
                    if (!isNaN(pctVal)) {
                        updateProgressBar(pctVal, currentActiveStep === "scan" ? "ASR Pass 1: SCAN..." : "ASR Pass 2: REFINE...");
                        // Also show percentage next to the specific ASR checklist step
                        if (currentActiveStep === "scan" && steps.scan) {
                            const pctEl = steps.scan.querySelector(".step-pct");
                            if (pctEl) pctEl.textContent = `(${pctVal.toFixed(0)}%)`;
                        } else if (currentActiveStep === "refine" && steps.refine) {
                            const pctEl = steps.refine.querySelector(".step-pct");
                            if (pctEl) pctEl.textContent = `(${pctVal.toFixed(0)}%)`;
                        }
                    }
                } catch (e) {
                    console.warn("Progress parse err:", e);
                }
            }
            
            appendLogLine(line, styleClass);
        };
        
        logEventSource.onerror = () => {
            console.log("Log EventSource closed or interrupted.");
        };
    }

    function updateProgressBar(percent, label) {
        const pct = Math.min(100, Math.max(0, percent));
        progressBar.style.width = `${pct}%`;
        progressPercent.textContent = `${pct.toFixed(0)}%`;
        if (label) {
            progressLabel.textContent = label;
        }
    }

    function appendLogLine(text, styleClass) {
        const lineEl = document.createElement("div");
        lineEl.className = styleClass;
        lineEl.textContent = text;
        terminalBody.appendChild(lineEl);
        terminalBody.scrollTop = terminalBody.scrollHeight; // Auto-scroll
    }

    // ── Checklist Management Helpers ──────────────────────────────────────────
    function resetSteps() {
        currentActiveStep = null;
        Object.keys(steps).forEach(key => {
            const step = steps[key];
            if (step) {
                step.className = "step pending";
                step.querySelector(".step-status").textContent = "○";
                step.querySelector(".step-time").textContent = "";
                const pctEl = step.querySelector(".step-pct");
                if (pctEl) pctEl.textContent = "";
            }
        });
    }

    function setStepActive(stepKey) {
        if (currentActiveStep && currentActiveStep !== stepKey) {
            setStepCompleted(currentActiveStep);
        }
        currentActiveStep = stepKey;
        if (steps[stepKey]) {
            steps[stepKey].className = "step running";
            steps[stepKey].querySelector(".step-status").textContent = "⚡";
        }
    }

    function setStepCompleted(stepKey, elapsedStr) {
        if (steps[stepKey]) {
            steps[stepKey].className = "step completed";
            steps[stepKey].querySelector(".step-status").textContent = "✓";
            if (elapsedStr) {
                steps[stepKey].querySelector(".step-time").textContent = elapsedStr;
            }
        }
    }

    function setStepFailed(stepKey) {
        if (steps[stepKey]) {
            steps[stepKey].className = "step failed";
            steps[stepKey].querySelector(".step-status").textContent = "✗";
        }
    }

    // ── Fetch Task Details upon Completion ────────────────────────────────────
    async function fetchTaskDetails(taskId) {
        try {
            const res = await fetch(`/api/tasks/${taskId}`);
            const task = await res.json();
            
            taskStatusBadge.className = `status-badge ${task.status}`;
            taskStatusBadge.textContent = task.status.toUpperCase();
            
            if (task.status === "completed") {
                updateProgressBar(100, "Localization complete!");
                // Complete all steps in the list
                Object.keys(steps).forEach(key => setStepCompleted(key));
                renderResult(task.result);
                loadHistory(); // Refresh recent sidebar
            } else if (task.status === "failed") {
                updateProgressBar(100, "Localization failed.");
                if (currentActiveStep) {
                    setStepFailed(currentActiveStep);
                }
                appendLogLine(`[Server] Pipeline failed: ${task.error}`, "error");
            }
            
        } catch (err) {
            console.error("Error fetching task details:", err);
        } finally {
            setFormDisabled(false);
        }
    }

    // ── Render Localization Result ───────────────────────────────────────────
    function renderResult(result) {
        if (!result) return;
        
        resultsPanel.classList.remove("hidden");
        
        // Load image frame
        resultFrame.src = result.frame_image_url || "";
        downloadFrameBtn.href = result.frame_image_url || "";
        
        // Confidence badge styling
        resultConfidence.textContent = result.confidence.toFixed(3);
        resultConfidence.className = `metric-value badge-${result.status}`;
        resultConfidenceStatus.textContent = `${result.status.replace("_", " ").toUpperCase()} CONFIDENCE`;
        
        // Text values
        resultTimestamp.textContent = result.timestamp_fmt;
        resultTimestampSec.textContent = `${result.timestamp_s.toFixed(3)}s`;
        resultFrameNumber.textContent = result.frame_number.toLocaleString();
        resultTargetText.textContent = `"${result.dialogue_text}"`;
        resultAsrText.textContent = `"${result.matched_text}"`;
        resultFrameWindow.textContent = `${result.vad_transition_s ? 'Onset ' + result.timestamp_s.toFixed(3) + 's' : result.timestamp_s.toFixed(3) + 's'}`;
        
        if (result.frame_image_url) {
            const filename = result.frame_image_url.split("/").pop();
            const frameNum = filename.replace("frame_", "").replace(".jpg", "");
            resultFrameWindow.textContent = `frame ${parseInt(frameNum).toLocaleString()}`;
        }

        // Populate confidence breakdown bars
        const tScore = result.text_score;
        const aScore = result.asr_quality;
        const vScore = result.vad_agreement;
        
        valText.textContent = tScore.toFixed(2);
        valAsr.textContent = aScore.toFixed(2);
        valVad.textContent = vScore.toFixed(2);
        
        // Scale segments based on weights (50%, 30%, 20%)
        weightText.style.width = `${tScore * 50}%`;
        weightAsr.style.width = `${aScore * 30}%`;
        weightVad.style.width = `${vScore * 20}%`;
        
        // Smooth scroll to results panel
        resultsPanel.scrollIntoView({ behavior: "smooth" });
    }

    // ── History List Sidebar ──────────────────────────────────────────────────
    async function loadHistory() {
        try {
            const res = await fetch("/api/tasks");
            const tasks = await res.json();
            
            if (tasks.length === 0) {
                historyList.innerHTML = '<div class="history-empty">No scans yet</div>';
                return;
            }
            
            historyList.innerHTML = "";
            tasks.forEach(t => {
                const item = document.createElement("div");
                item.className = "history-item";
                if (t.task_id === activeTaskId) {
                    item.className += " active";
                }
                
                const statusColor = t.status === "completed" ? "var(--success)" : (t.status === "failed" ? "var(--danger)" : "var(--warning)");
                
                item.innerHTML = `
                    <div class="item-header">
                        <span class="target" title="${t.target}">${t.target}</span>
                        <span class="status-dot" style="width: 8px; height: 8px; border-radius: 50%; background: ${statusColor};"></span>
                    </div>
                    <div class="url" title="${t.url}">${t.url}</div>
                `;
                
                item.addEventListener("click", () => {
                    selectHistoryItem(t.task_id);
                });
                
                historyList.appendChild(item);
            });
            
        } catch (err) {
            console.error("Error loading task history:", err);
        }
    }

    async function selectHistoryItem(taskId) {
        activeTaskId = taskId;
        
        // Highlight active item
        const items = historyList.querySelectorAll(".history-item");
        items.forEach((item, idx) => {
            item.classList.remove("active");
        });
        
        // Close running streams
        if (logEventSource) {
            logEventSource.close();
        }
        
        // Show panels and fetch detailed info
        processingPanel.classList.remove("hidden");
        resultsPanel.classList.add("hidden");
        setFormDisabled(true);
        
        updateProgressBar(0, "Fetching details...");
        terminalBody.innerHTML = '<div class="log-line system">Loading logs from server database...</div>';
        resetSteps();
        
        try {
            const res = await fetch(`/api/tasks/${taskId}`);
            const task = await res.json();
            
            taskStatusBadge.className = `status-badge ${task.status}`;
            taskStatusBadge.textContent = task.status.toUpperCase();
            
            // Populate terminal logs and check step highlights
            terminalBody.innerHTML = "";
            task.logs.forEach(line => {
                let styleClass = "log-line";
                if (line.includes("[SCAN]")) styleClass = "log-line scan";
                else if (line.includes("[REFINE]")) styleClass = "log-line refine";
                else if (line.includes("Error:") || line.includes("failed")) styleClass = "log-line error";
                else if (line.includes("complete") || line.includes("Success")) styleClass = "log-line success";
                else if (line.startsWith("[Server]")) styleClass = "log-line system";
                
                // Replay step states for history loading
                if (line.includes("Stage: Video Download")) setStepActive("download");
                else if (line.includes("Stage: Audio Extraction")) { setStepCompleted("download"); setStepActive("audio"); }
                else if (line.includes("Stage: VAD")) { setStepCompleted("audio"); setStepActive("vad"); }
                else if (line.includes("Stage: ASR Phase 1: SCAN")) { setStepCompleted("vad"); setStepActive("scan"); }
                else if (line.includes("Stage: ASR Phase 2: REFINE")) { setStepCompleted("scan"); setStepActive("refine"); }
                else if (line.includes("Stage: Forced Alignment")) { setStepCompleted("refine"); setStepActive("align"); }
                else if (line.includes("Stage: Frame PTS Read") || line.includes("Stage: Fuzzy Phrase Match (refine)")) { setStepCompleted("align"); setStepActive("extract"); }
                else if (line.includes("Pipeline complete")) { setStepCompleted("extract"); }

                if (line.includes("done in") && currentActiveStep) {
                    try {
                        const duration = line.split("done in")[1].trim().replace("?", "").replace("…", "");
                        setStepCompleted(currentActiveStep, duration);
                    } catch (err) {
                        setStepCompleted(currentActiveStep);
                    }
                }
                
                appendLogLine(line, styleClass);
            });
            
            if (task.status === "completed") {
                updateProgressBar(100, "Completed successfully!");
                Object.keys(steps).forEach(key => setStepCompleted(key));
                renderResult(task.result);
            } else if (task.status === "failed") {
                updateProgressBar(100, "Task failed.");
                if (currentActiveStep) {
                    setStepFailed(currentActiveStep);
                }
                appendLogLine(`[Server] Pipeline error description: ${task.error}`, "error");
            } else {
                // Resume live log streaming if task is still running or pending
                startLogStream(taskId);
            }
            
            // Re-render active item class selection in history sidebar
            loadHistory();
            
        } catch (err) {
            console.error("Error loading history item:", err);
            appendLogLine(`[Server] Load failed: ${err.message}`, "error");
        } finally {
            setFormDisabled(false);
        }
    }

    function setFormDisabled(disabled) {
        const inputs = locateForm.querySelectorAll("input");
        inputs.forEach(i => i.disabled = disabled);
        submitBtn.disabled = disabled;
    }
});
