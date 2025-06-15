// script.js

let mediaRecorder;
let audioChunks = [];
const RECORD_DURATION_MS = 4000;

let lastPlayedFeedbackText = "";
let lastDisplayedVoiceCommandText = "";
let lastDisplayedNluStatus = "";       // Track NLU status displayed on UI
let lastDisplayedNluResultText = ""; // Track NLU result text displayed on UI


function sendCommandToBackend(command, payload = {}) {
    addToLog('发送指令到后端: ' + command + (Object.keys(payload).length ? ' Payload: ' + JSON.stringify(payload) : ''));

    if (command === 'process_voice_text' || command === 'pick_object') {
        lastPlayedFeedbackText = "";
        lastDisplayedNluStatus = "正在处理..."; // Set to an intermediate state
        lastDisplayedNluResultText = "正在处理..."; // Set to an intermediate state
    }

    // For simulated text, update UI immediately for responsiveness
    if (command === 'process_voice_text' && payload.text) {
        const recognizedTextEl = document.getElementById('recognizedText');
        if (recognizedTextEl) {
            recognizedTextEl.textContent = payload.text + " (模拟...)";
            lastDisplayedVoiceCommandText = recognizedTextEl.textContent;
        }
        const voiceFeedbackEl = document.getElementById('voiceFeedback');
        if (voiceFeedbackEl) voiceFeedbackEl.textContent = "正在理解模拟指令...";
        const nluStatusEl = document.getElementById('nluStatus');
        if (nluStatusEl) nluStatusEl.textContent = "正在理解模拟指令...";
        const nluResultTextEl = document.getElementById('nluResultText');
        if (nluResultTextEl) nluResultTextEl.textContent = "处理中...";
    }


    fetch('/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', },
        body: JSON.stringify({ command: command, payload: payload }),
    })
        .then(response => {
            const responseStatusHttp = response.status;
            const responseOkHttp = response.ok;
            return response.json().then(data => ({
                ok: responseOkHttp,
                statusHttp: responseStatusHttp,
                body: data
            }));
        })
        .then(result => {
            const data = result.body;
            addToLog('后端响应: ' + JSON.stringify(data));

            if (!result.ok) {
                const errorMessage = data.message || `HTTP error ${result.statusHttp}`;
                throw new Error(errorMessage);
            }

            // Backend /control now returns a message which could be the NLU feedback.
            // updateSystemMessage is for general system messages.
            // The main NLU feedback for TTS will be handled by /status polling.
            if (data.message) {
                updateSystemMessage(data.message, data.status === 'error' ? 'error' : 'info');
            }

            updateFullStatus(); // Crucial: fetch latest full state after command.
        })
        .catch((error) => {
            const errorMessage = error.message || '与后端通信发生未知错误。';
            addToLog('指令错误: ' + errorMessage);
            updateSystemMessage('与后端通信失败: ' + errorMessage, 'error');
            console.error('Error during command fetch:', error);
        });
}

function isFinalNluStatus(nluStatus) {
    if (!nluStatus) return false;
    const intermediatePatterns = ["正在识别", "正在理解", "音频已接收", "待处理", "处理中", "上传识别中", "待识别"];
    return !intermediatePatterns.some(pattern => nluStatus.includes(pattern));
}

function shouldPlayFeedback(feedbackText, nluStatus) {
    // ... (此函数逻辑保持不变)
    if (!feedbackText || feedbackText.trim() === "" || feedbackText === "待处理..." || feedbackText === "处理中...") {
        return false;
    }
    const nonPlayableFeedbackPrefixes = [
        "理解结果格式错误:", "抱歉，", "网络请求失败", "发生未知内部错误",
        "错误:", "配置错误", "NLU服务未返回有效内容", "理解成功，但无反馈文本。"
    ];
    if (nonPlayableFeedbackPrefixes.some(prefix => feedbackText.startsWith(prefix))) {
        return false;
    }
    const playableFinalNluStatuses = [
        "指令理解成功",
        "UI拾取",
        "未在画面中找到",
        "NLU结果: 未指定拾取物体"
    ];
    const errorOrIntermediateNluStatuses = [
        "ASR失败", "ASR网络错误", "ASR响应格式错误", "ASR未知错误",
        "NLU失败:", "NLU配置错误", "NLU网络错误", "NLU结果格式错误",
        "NLU无有效内容", "NLU响应格式错误", "NLU未知错误",
        "待识别...", "待理解...", "就绪",
        "正在识别语音...", "正在理解指令...", "正在调用NLU服务...",
        "音频已接收，准备处理...", "上传识别中...",
        "操作已取消", "用户触发复位"
    ];

    if (nluStatus && errorOrIntermediateNluStatuses.some(s => nluStatus.startsWith(s))) {
        return false;
    }
    if (nluStatus && playableFinalNluStatuses.some(s => nluStatus.startsWith(s))) {
        return true;
    }
    return false;
}



function playSynthesizedAudio(text) {
    // ... (playSynthesizedAudio 函数保持不变)
    if (!text || text.trim() === "") return;
    addToLog('请求语音合成: ' + text.substring(0, 50) + "...");
    fetch(`/get_tts_audio?text=${encodeURIComponent(text)}`)
        .then(response => {
            if (response.ok && response.headers.get('content-type') && response.headers.get('content-type').includes('audio/mp3')) {
                return response.blob();
            }
            return response.json().then(errData => {
                throw new Error(errData.message || `TTS audio fetch failed with status ${response.status}`);
            });
        })
        .then(audioBlob => {
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            audio.play()
                .then(() => { addToLog('语音合成播放已启动.'); })
                .catch(e => { addToLog('语音播放失败: ' + e.message); console.error('Audio playback error:', e); });
            audio.onended = () => { URL.revokeObjectURL(audioUrl); addToLog('语音播放完毕.'); };
            audio.onerror = (e) => {
                addToLog('音频元素错误: ' + (e.target.error ? e.target.error.message : 'Unknown audio error'));
                console.error('Audio element error:', e.target.error);
                URL.revokeObjectURL(audioUrl);
            };
        })
        .catch(error => {
            addToLog('语音合成或播放错误: ' + error.message);
            console.error('Error fetching/playing TTS audio:', error);
        });
}

function updateFullStatus() {
    fetch('/status')
        .then(response => {
            if (!response.ok) { throw new Error(`HTTP error ${response.status}`); }
            return response.json();
        })
        .then(data => {
            // console.log("STATUS_RECEIVED:", JSON.stringify(data)); 
            // console.log("LAST_PLAYED_FEEDBACK:", lastPlayedFeedbackText, "LAST_PROCESSED_NLU_STATUS:", lastProcessedNluStatus);

            // --- UI Updates ---
            document.getElementById('armMainState').textContent = data.arm_state || '未知';
            const lightStatusText = data.stm32_light_status || '未知';
            document.getElementById('stm32LightStatusText').textContent = lightStatusText;
            updateLedIndicators(lightStatusText);
            const trajIndex = data.current_trajectory_index !== undefined ? data.current_trajectory_index : -1;
            const totalPoints = data.total_trajectory_points || 0;
            if (data.arm_state === 'RUNNING_TRAJECTORY' || data.arm_state === 'PAUSED_TRAJECTORY') {
                document.getElementById('trajectoryInfo').textContent = `${trajIndex + 1} / ${totalPoints}`;
            } else {
                document.getElementById('trajectoryInfo').textContent = 'N/A';
            }
            document.getElementById('armCoords').textContent = data.arm_coords || '[X, Y, Z, A, B, C]';
            document.getElementById('pumpStatus').textContent = data.pump_status || 'OFF';
            document.getElementById('pumpAngleDisplay').textContent = (data.pump_angle !== undefined ? data.pump_angle : '0') + '°';

            const recTextEl = document.getElementById('recognizedText');
            if (recTextEl && data.voice_command_text && data.voice_command_text !== lastDisplayedVoiceCommandText) {
                const clientIntermediateMessages = ["录音中...", "请求权限...", "上传识别中...", "模拟处理中..."];
                if (!clientIntermediateMessages.includes(recTextEl.textContent) ||
                    (clientIntermediateMessages.includes(recTextEl.textContent) && !clientIntermediateMessages.includes(data.voice_command_text) && data.voice_command_text !== "待识别...")) {
                    recTextEl.textContent = data.voice_command_text;
                    lastDisplayedVoiceCommandText = data.voice_command_text;
                }
            }

            const nluStatusEl = document.getElementById('nluStatus');
            if (nluStatusEl && data.nlu_status) nluStatusEl.textContent = data.nlu_status;

            const nluResultTextEl = document.getElementById('nluResultText');
            if (nluResultTextEl && data.nlu_result_text) nluResultTextEl.textContent = data.nlu_result_text;

            const voiceFeedbackEl = document.getElementById('voiceFeedback');
            if (voiceFeedbackEl && data.nlu_result_text) {
                if (data.nlu_result_text !== "待处理..." && data.nlu_result_text !== "处理中...") {
                    voiceFeedbackEl.textContent = data.nlu_result_text;
                } else if (data.nlu_status && !data.nlu_status.startsWith("正在") && data.nlu_status !== "音频已接收，准备处理..." && data.nlu_status !== "就绪") {
                    voiceFeedbackEl.textContent = data.nlu_status; // 如果nlu_result是占位符，但nlu_status是最终状态，显示nlu_status
                }
            }

            // --- 更新 "系统消息与提示" (systemMessages) ---
            // 我们希望 systemMessages 显示一个能反映当前最重要用户反馈的状态
            // 通常，如果NLU成功，nlu_result_text (feedback_text) 是最好的。
            // 如果NLU有明确的错误或状态，nlu_status 也可以。
            // 避免一直显示 "音频已上传..."
            const systemMessagesEl = document.getElementById('systemMessages');
            if (systemMessagesEl) {
                let messageForSystem = "就绪，等待操作或语音指令。"; // 默认消息
                let messageType = 'info';

                if (data.nlu_status) {
                    // 优先使用 nlu_result_text 如果它不是占位符且 nlu_status 是成功的
                    if (data.nlu_result_text && data.nlu_result_text !== "待处理..." && data.nlu_result_text !== "处理中..." &&
                        (data.nlu_status.includes("指令理解成功") || data.nlu_status.includes("UI拾取") || data.nlu_status.includes("未在画面中找到"))) {
                        messageForSystem = data.nlu_result_text;
                        messageType = 'success'; // 或 'info'
                    }
                    // 否则，如果nlu_status本身是一个有意义的最终用户提示（不是中间过程）
                    else if (!data.nlu_status.startsWith("正在") &&
                        data.nlu_status !== "音频已接收，准备处理..." &&
                        data.nlu_status !== "就绪" &&
                        data.nlu_status !== "指令理解成功" && // 因为上面的条件已经处理了这种情况下的nlu_result_text
                        data.nlu_status !== "UI拾取" &&
                        data.nlu_status !== "待识别...") {
                        messageForSystem = data.nlu_status;
                        if (data.nlu_status.includes("失败") || data.nlu_status.includes("错误")) {
                            messageType = 'error';
                        } else {
                            messageType = 'info';
                        }
                    }
                    // 如果 nlu_status 是中间过程状态，systemMessages 可以保持更通用的提示，或者由其他逻辑更新
                    // 例如，triggerRecordAudio 会临时更新它
                }

                // 只有当计算出的 messageForSystem 与当前 systemMessages 不同，或者不是录音上传后的那个特定消息时，才更新
                // 以免频繁闪烁或覆盖掉 triggerRecordAudio 中设置的即时消息
                const currentSystemMessage = systemMessagesEl.textContent;
                const uploadInProgressMessage = "音频已上传，等待识别结果。"; // 这是 triggerRecordAudio 设置的
                const uploadBackendMessage = "音频已上传，正在后台处理..."; // 这是后端 /upload_voice_audio 返回的

                if (messageForSystem !== currentSystemMessage &&
                    currentSystemMessage !== uploadInProgressMessage && // 不要覆盖掉这个
                    currentSystemMessage !== uploadBackendMessage &&
                    !currentSystemMessage.startsWith("正在录音") && // 不要覆盖客户端设置的录音过程消息
                    !currentSystemMessage.startsWith("正在请求麦克风权限")
                ) {
                    // updateSystemMessage(messageForSystem, messageType); // 这行会导致循环，因为updateSystemMessage会改变textContent
                    // 直接设置，避免循环调用
                    systemMessagesEl.textContent = messageForSystem;
                    const colors = {
                        info: { bg: '#e2f3ff', text: '#0c5460' },
                        error: { bg: '#f8d7da', text: '#721c24' },
                        success: { bg: '#d4edda', text: '#155724' }
                    };
                    const effectiveType = colors[messageType] ? messageType : 'info';
                    systemMessagesEl.style.backgroundColor = colors[effectiveType].bg;
                    systemMessagesEl.style.color = colors[effectiveType].text;
                }
            }


            // --- TTS 触发逻辑 ---
            const feedbackTextToPlay = data.nlu_result_text || "";
            const currentNluStatus = data.nlu_status || "";
            let playNow = false;

            if (feedbackTextToPlay && feedbackTextToPlay !== "待处理..." && feedbackTextToPlay !== "处理中...") {
                if (feedbackTextToPlay !== lastPlayedFeedbackText) {
                    if (shouldPlayFeedback(feedbackTextToPlay, currentNluStatus)) {
                        playNow = true;
                    }
                } else {
                    if (currentNluStatus !== lastProcessedNluStatus &&
                        shouldPlayFeedback(feedbackTextToPlay, currentNluStatus) &&
                        !shouldPlayFeedback(feedbackTextToPlay, lastProcessedNluStatus)) {
                        playNow = true;
                    }
                }
            }
            if (playNow) {
                playSynthesizedAudio(feedbackTextToPlay);
                lastPlayedFeedbackText = feedbackTextToPlay;
            }
            lastProcessedNluStatus = currentNluStatus;

            // ... (物体列表和目标信息更新保持不变)
            const objectsListEl = document.getElementById('recognizedObjectsList');
            objectsListEl.innerHTML = '';
            if (data.detected_objects && data.detected_objects.length > 0) {
                data.detected_objects.forEach(obj => {
                    const li = document.createElement('li');
                    li.innerHTML = `${obj.name} (置信度: ${(obj.confidence * 100).toFixed(1)}%) - 中心: [${obj.box_center_x},${obj.box_center_y}] 
                                  <button onclick="requestPickObject('${obj.id}', ${obj.box_center_x}, ${obj.box_center_y}, '${obj.name}')">拾取</button>`;
                    objectsListEl.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = '当前未检测到可识别物体。';
                objectsListEl.appendChild(li);
            }
            document.getElementById('targetObjectInfo').textContent = data.target_object_info || '未指定';
            document.getElementById('targetLocationInfo').textContent = data.target_location_info || '未指定';
        })
        .catch((error) => {
            addToLog('获取状态错误: ' + error.message);
            // 不要在这里调用 updateSystemMessage, 避免覆盖更具体的错误
            // updateSystemMessage('获取系统状态失败: ' + error.message, 'error');
            console.error('Error fetching status:', error);
        });
}

function requestPickObject(objectId, pixelX, pixelY, objectName) {
    addToLog(`请求拾取物体: ${objectName} (ID: ${objectId}) at pixel (${pixelX}, ${pixelY})`);
    sendCommandToBackend('pick_object', {
        id: objectId,
        pixel_x: pixelX,
        pixel_y: pixelY,
        name: objectName
    });
    document.getElementById('targetObjectInfo').textContent = `${objectName} (像素: ${pixelX},${pixelY})`;
}

// ... (updateLedIndicators, addToLog, updateSystemMessage 保持不变)
function updateLedIndicators(statusText) {
    const leds = {
        red: document.getElementById('ledRed'),
        green: document.getElementById('ledGreen'),
        yellow: document.getElementById('ledYellow')
    };
    for (let color in leds) {
        if (leds[color]) {
            leds[color].classList.remove('on');
        }
    }
    if (!leds.red || !leds.green || !leds.yellow) return;

    if (statusText === "RED_ON") leds.red.classList.add('on');
    else if (statusText === "GREEN_ON") leds.green.classList.add('on');
    else if (statusText === "YELLOW_ON") leds.yellow.classList.add('on');
}

function addToLog(message) {
    const logArea = document.getElementById('logArea');
    if (!logArea) return;
    const timestamp = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const currentContent = logArea.value;
    const newContent = `[${timestamp}] ${message}\n`;
    logArea.value = newContent + currentContent;
    const maxLogLength = 50000;
    if (logArea.value.length > maxLogLength) {
        logArea.value = logArea.value.substring(0, maxLogLength);
    }
}

function updateSystemMessage(message, type = 'info') {
    const msgArea = document.getElementById('systemMessages');
    if (!msgArea) return;
    msgArea.textContent = message;
    const colors = {
        info: { bg: '#e2f3ff', text: '#0c5460' },
        error: { bg: '#f8d7da', text: '#721c24' },
        success: { bg: '#d4edda', text: '#155724' }
    };
    const effectiveType = colors[type] ? type : 'info';
    msgArea.style.backgroundColor = colors[effectiveType].bg;
    msgArea.style.color = colors[effectiveType].text;
}


function sendVoiceCommand() {
    const commandInput = document.getElementById('voiceCommandInput');
    if (!commandInput) return;
    const commandText = commandInput.value;

    if (!commandText.trim()) {
        updateSystemMessage('请输入模拟语音指令。', 'error');
        return;
    }
    addToLog('模拟语音指令: ' + commandText);
    // sendCommandToBackend 内部会为 'process_voice_text' 重置 lastPlayedFeedbackText等
    // 并更新UI初始状态
    sendCommandToBackend('process_voice_text', { text: commandText });
}

async function triggerRecordAudio() {
    addToLog('请求客户端录音...');
    lastPlayedFeedbackText = "";
    lastDisplayedVoiceCommandText = "";
    lastDisplayedNluStatus = "";
    lastDisplayedNluResultText = "";


    const recognizedTextEl = document.getElementById('recognizedText');
    const voiceFeedbackEl = document.getElementById('voiceFeedback');
    const nluStatusEl = document.getElementById('nluStatus');
    const nluResultTextEl = document.getElementById('nluResultText');


    if (recognizedTextEl) recognizedTextEl.textContent = "请求权限...";
    lastDisplayedVoiceCommandText = "请求权限...";
    updateSystemMessage('正在请求麦克风权限...', 'info'); // General message
    if (voiceFeedbackEl) voiceFeedbackEl.textContent = "请授权麦克风..."; // Main feedback area
    if (nluStatusEl) nluStatusEl.textContent = "等待授权...";
    if (nluResultTextEl) nluResultTextEl.textContent = "...";

    // ... (其余录音逻辑保持不变)
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const errorMsg = '错误: 浏览器不支持录音功能 (getUserMedia API)。';
        addToLog(errorMsg);
        updateSystemMessage(errorMsg, 'error');
        if (recognizedTextEl) { recognizedTextEl.textContent = "浏览器不支持"; lastDisplayedVoiceCommandText = "浏览器不支持"; }
        if (voiceFeedbackEl) voiceFeedbackEl.textContent = "录音功能不可用";
        if (nluStatusEl) nluStatusEl.textContent = "录音功能不可用";
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        let mimeTypeToUse = 'audio/webm';
        if (MediaRecorder.isTypeSupported('audio/wav; codecs=MS_PCM')) {
            mimeTypeToUse = 'audio/wav; codecs=MS_PCM';
        } else if (MediaRecorder.isTypeSupported('audio/wav')) {
            mimeTypeToUse = 'audio/wav';
        } else if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
            mimeTypeToUse = 'audio/webm;codecs=opus';
        } else if (!MediaRecorder.isTypeSupported(mimeTypeToUse)) {
            if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                mimeTypeToUse = 'audio/ogg;codecs=opus';
            } else {
                addToLog('警告: 常见的期望MIME类型不被支持，使用浏览器默认。');
            }
        }
        mediaRecorder = new MediaRecorder(stream, { mimeType: mimeTypeToUse });
        addToLog(`麦克风权限已获取，开始录音 (MIME: ${mimeTypeToUse})...`);
        if (recognizedTextEl) { recognizedTextEl.textContent = "录音中..."; lastDisplayedVoiceCommandText = "录音中..."; }
        updateSystemMessage(`正在录音 (${RECORD_DURATION_MS / 1000}秒)...`, 'info');
        if (voiceFeedbackEl) voiceFeedbackEl.textContent = "请说话...";
        if (nluStatusEl) nluStatusEl.textContent = "正在录音...";

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        mediaRecorder.onstop = () => {
            stream.getTracks().forEach(track => track.stop());
            if (audioChunks.length === 0) {
                addToLog('录音失败: 没有录制到音频数据。');
                updateSystemMessage('录音失败: 未录制到音频数据。', 'error');
                if (recognizedTextEl) { recognizedTextEl.textContent = "录音数据为空"; lastDisplayedVoiceCommandText = "录音数据为空"; }
                if (voiceFeedbackEl) voiceFeedbackEl.textContent = "没有录到声音，请重试。";
                if (nluStatusEl) nluStatusEl.textContent = "录音数据为空";
                return;
            }
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
            audioChunks = [];
            addToLog(`录音完成. Blob type: ${audioBlob.type}, Blob size: ${audioBlob.size} bytes. 准备上传...`);
            if (recognizedTextEl) { recognizedTextEl.textContent = "上传识别中..."; lastDisplayedVoiceCommandText = "上传识别中..."; }
            updateSystemMessage('录音完成，正在上传和识别...', 'info');
            if (voiceFeedbackEl) voiceFeedbackEl.textContent = "处理中，请稍候...";
            if (nluStatusEl) nluStatusEl.textContent = "音频已上传，等待后端处理...";


            const formData = new FormData();
            let fileExtension = ".webm";
            if (audioBlob.type.includes("wav")) fileExtension = ".wav";
            else if (audioBlob.type.includes("ogg")) fileExtension = ".ogg";
            formData.append('audio_data', audioBlob, 'recorded_audio' + fileExtension);
            fetch('/upload_voice_audio', {
                method: 'POST',
                body: formData
            })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(errData => {
                            throw new Error(errData.message || `HTTP error ${response.status}`);
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    addToLog('音频上传成功，后端响应: ' + JSON.stringify(data));
                    updateSystemMessage(data.message || '音频已上传，等待识别结果。', 'info');
                    // 后端已收到上传请求，`/status` 轮询将获取后续的ASR/NLU结果并触发TTS
                    updateFullStatus();
                })
                .catch(error => {
                    const errorMessage = error.message || '音频上传或处理失败。';
                    addToLog('音频上传错误: ' + errorMessage);
                    updateSystemMessage('音频上传错误: ' + errorMessage, 'error');
                    if (recognizedTextEl) { recognizedTextEl.textContent = "上传失败"; lastDisplayedVoiceCommandText = "上传失败"; }
                    if (voiceFeedbackEl) voiceFeedbackEl.textContent = "处理失败，请检查网络或重试。";
                    if (nluStatusEl) nluStatusEl.textContent = "上传失败";
                    console.error('Error uploading audio:', error);
                });
        };
        mediaRecorder.start();
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === "recording") {
                mediaRecorder.stop();
                addToLog('录音时间到，自动停止。');
            }
        }, RECORD_DURATION_MS);
    } catch (err) {
        const errorMsg = '麦克风访问被拒绝或出错: ' + err.name + " - " + err.message;
        addToLog(errorMsg);
        updateSystemMessage(errorMsg, 'error');
        if (recognizedTextEl) { recognizedTextEl.textContent = "麦克风错误"; lastDisplayedVoiceCommandText = "麦克风错误"; }
        if (voiceFeedbackEl) voiceFeedbackEl.textContent = "无法使用麦克风";
        if (nluStatusEl) nluStatusEl.textContent = "麦克风错误";
        console.error('Error accessing microphone:', err);
    }
}

// ... (setPumpAngle 和 DOMContentLoaded 保持不变)
function setPumpAngle() {
    const angleInput = document.getElementById('pumpAngleInput');
    if (!angleInput) return;
    const angle = parseInt(angleInput.value);

    if (isNaN(angle) || angle < 0 || angle > 180) {
        updateSystemMessage('请输入有效的旋转角度 (0 到 180)。', 'error');
        angleInput.focus();
        return;
    }
    addToLog('设定旋转头角度: ' + angle + '°');
    sendCommandToBackend('pump_angle_set', { angle: angle });
}

document.addEventListener('DOMContentLoaded', function () {
    const logAreaDiv = document.getElementById('logArea');
    if (logAreaDiv && logAreaDiv.tagName !== 'TEXTAREA') {
        const logAreaTextarea = document.createElement('textarea');
        logAreaTextarea.id = 'logArea';
        logAreaTextarea.readOnly = true;
        logAreaTextarea.style.cssText = window.getComputedStyle(logAreaDiv).cssText;
        logAreaTextarea.value = logAreaDiv.textContent || logAreaDiv.innerText || "";
        logAreaDiv.parentNode.replaceChild(logAreaTextarea, logAreaDiv);
    } else if (logAreaDiv) {
        logAreaDiv.value = "";
    }

    updateFullStatus();
    setInterval(updateFullStatus, 1000); // 缩短轮询间隔，以便更快响应后端状态变化

    addToLog("Web UI 已加载，开始与后端通信...");
    updateSystemMessage("就绪，等待操作或语音指令。");

    const voiceCommandInput = document.getElementById('voiceCommandInput');
    if (voiceCommandInput) {
        voiceCommandInput.addEventListener('keypress', function (event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                sendVoiceCommand();
            }
        });
    }
});