function sendCommandToBackend(command, payload = {}) {
    addToLog('发送指令到后端: ' + command + (Object.keys(payload).length ? ' Payload: ' + JSON.stringify(payload) : ''));
    fetch('/control', { // 假设后端API端点为 /control
        method: 'POST',
        headers: { 'Content-Type': 'application/json', },
        body: JSON.stringify({ command: command, payload: payload }),
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
            addToLog('后端响应: ' + JSON.stringify(data));
            if (data.message) {
                updateSystemMessage(data.message, data.status === 'error' ? 'error' : 'info');
            }
            updateFullStatus(); // 更新状态以反映后端变化
        })
        .catch((error) => {
            const errorMessage = error.message || '与后端通信发生未知错误。';
            addToLog('指令错误: ' + errorMessage);
            updateSystemMessage('与后端通信失败: ' + errorMessage, 'error');
            console.error('Error during command fetch:', error);
        });
}

function updateFullStatus() {
    fetch('/status') // 假设后端API端点为 /status
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
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
            // document.getElementById('armJoints').textContent = data.arm_joints || '[J1, J2, J3, J4, J5, J6]';

            document.getElementById('pumpStatus').textContent = data.pump_status || 'OFF';
            document.getElementById('pumpAngleDisplay').textContent = (data.pump_angle !== undefined ? data.pump_angle : '0') + '°';

            // 更新语音识别和反馈 (如果后端状态包含这些)
            if (data.recognized_text) {
                document.getElementById('recognizedText').textContent = data.recognized_text;
            }
            if (data.voice_feedback) {
                document.getElementById('voiceFeedback').textContent = data.voice_feedback;
            }

            // --- 动态更新物体识别列表 ---
            const objectsListEl = document.getElementById('recognizedObjectsList');
            objectsListEl.innerHTML = ''; // 清空现有列表项
            if (data.detected_objects && data.detected_objects.length > 0) {
                data.detected_objects.forEach(obj => {
                    const li = document.createElement('li');
                    // 注意：obj.box_center_x 和 obj.box_center_y 是像素坐标
                    // obj.id 应该是后端为每个检测到的物体生成的唯一ID
                    li.innerHTML = `${obj.name} (置信度: ${(obj.confidence * 100).toFixed(1)}%) - 中心: [${obj.box_center_x},${obj.box_center_y}] 
                                  <button onclick="requestPickObject('${obj.id}', ${obj.box_center_x}, ${obj.box_center_y}, '${obj.name}')">拾取</button>`;
                    objectsListEl.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = '当前未检测到可识别物体。';
                objectsListEl.appendChild(li);
            }
            
            // 更新任务目标信息 (如果后端状态包含这些)
            document.getElementById('targetObjectInfo').textContent = data.target_object_info || '未指定';
            document.getElementById('targetLocationInfo').textContent = data.target_location_info || '未指定';

        })
        .catch((error) => {
            addToLog('获取状态错误: ' + error.message);
            updateSystemMessage('获取系统状态失败: ' + error.message, 'error');
            console.error('Error fetching status:', error);
        });
}

function requestPickObject(objectId, pixelX, pixelY, objectName) {
    addToLog(`请求拾取物体: ${objectName} (ID: ${objectId}) at pixel (${pixelX}, ${pixelY})`);
    // 发送包含像素坐标和物体ID/名称的拾取指令
    sendCommandToBackend('pick_object', {
        id: objectId, // 后端可能需要这个ID来查找物体的3D坐标等信息
        pixel_x: pixelX,
        pixel_y: pixelY,
        name: objectName
    });
    updateSystemMessage(`拾取指令已发送: ${objectName}`, 'info');
    document.getElementById('targetObjectInfo').textContent = `${objectName} (像素: ${pixelX},${pixelY})`; // 临时更新UI
}

function updateLedIndicators(statusText) {
    const leds = {
        red: document.getElementById('ledRed'),
        green: document.getElementById('ledGreen'),
        yellow: document.getElementById('ledYellow')
    };
    for (let color in leds) {
        if (leds[color]) { // 确保元素存在
            leds[color].classList.remove('on');
        }
    }
    if (!leds.red || !leds.green || !leds.yellow) return; // 如果任何LED元素未找到，则退出

    if (statusText === "RED_ON") leds.red.classList.add('on');
    else if (statusText === "GREEN_ON") leds.green.classList.add('on');
    else if (statusText === "YELLOW_ON") leds.yellow.classList.add('on');
}

function addToLog(message) {
    const logArea = document.getElementById('logArea');
    if (!logArea) return; // 确保元素存在
    const timestamp = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    // 如果 logArea 是 div
    // logArea.innerHTML += `[${timestamp}] ${message}\n`;
    
    // 如果 logArea 是 textarea (根据原始HTML中DOM操作，它被改成了textarea)
    logArea.value += `[${timestamp}] ${message}\n`;
    logArea.scrollTop = logArea.scrollHeight;
}

function updateSystemMessage(message, type = 'info') {
    const msgArea = document.getElementById('systemMessages');
    if (!msgArea) return; // 确保元素存在
    msgArea.textContent = message;
    const colors = {
        info: { bg: '#e2f3ff', text: '#0c5460' },
        error: { bg: '#f8d7da', text: '#721c24' },
        success: { bg: '#d4edda', text: '#155724' }
    };
    msgArea.style.backgroundColor = colors[type] ? colors[type].bg : colors.info.bg;
    msgArea.style.color = colors[type] ? colors[type].text : colors.info.text;
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
    sendCommandToBackend('process_voice_text', { text: commandText });
    
    const recognizedTextEl = document.getElementById('recognizedText');
    if (recognizedTextEl) {
        recognizedTextEl.textContent = commandText + " (等待后端解析...)";
    }
    updateSystemMessage('模拟语音指令已发送处理...', 'info');
    commandInput.value = ''; // 清空输入框
}

function triggerRecordAudio() {
    addToLog('请求硬件录音...');
    sendCommandToBackend('start_hardware_voice_recognition');
    updateSystemMessage('硬件录音请求已发送...', 'info');
    const recognizedTextEl = document.getElementById('recognizedText');
    if (recognizedTextEl) {
        recognizedTextEl.textContent = "正在录音/识别...";
    }
}

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

// DOMContentLoaded 事件监听器确保在执行脚本前HTML已完全加载和解析
document.addEventListener('DOMContentLoaded', function () {
    // 将日志区域改为 textarea 以支持 .value 和滚动，这一步在原始HTML内联脚本中已完成。
    // 如果 #logArea 初始就是 textarea，则无需此操作。
    // 如果初始是 div，并且希望在这里确保它是 textarea，可以保留或调整这里的逻辑。
    const logAreaDiv = document.getElementById('logArea');
    if (logAreaDiv && logAreaDiv.tagName !== 'TEXTAREA') { // 检查是否已经是textarea
        const logAreaTextarea = document.createElement('textarea');
        logAreaTextarea.id = 'logArea';
        logAreaTextarea.readOnly = true;
        // 复制原始div的样式和内容
        logAreaTextarea.style.cssText = window.getComputedStyle(logAreaDiv).cssText;
        logAreaTextarea.style.whiteSpace = 'pre-wrap'; // 确保textarea也换行
        logAreaTextarea.value = logAreaDiv.textContent || logAreaDiv.innerText || ""; // 迁移内容
        
        logAreaDiv.parentNode.replaceChild(logAreaTextarea, logAreaDiv);
    }


    updateFullStatus(); // 页面加载时获取一次初始状态
    setInterval(updateFullStatus, 2000); // 每2秒更新一次状态

    addToLog("Web UI 已加载，开始与后端通信...");
    updateSystemMessage("就绪，等待操作或语音指令。");

    // 为模拟语音输入框添加回车键发送功能
    const voiceCommandInput = document.getElementById('voiceCommandInput');
    if (voiceCommandInput) {
        voiceCommandInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault(); // 阻止默认的回车行为（如表单提交）
                sendVoiceCommand();
            }
        });
    }
});