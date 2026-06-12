/**
 * script.js — GuideX Core front-end logic (Clean active implementation)
 */

let isNavigating = false;
let currentLanguage = 'en';
let currentLocation = null;
let recognitionActive = false;
let registeredFaces = [];
let alertInterval = null;

const translations = {
    en: {
        ready: "System ready",
        cameraActive: "Camera active",
        start: "Starting navigation",
        stop: "Navigation stopped",
        listening: "Listening...",
        processing: "Processing...",
        speaking: "Speaking...",
        textFound: "Text found",
        noText: "No text detected",
        navigating: "Navigating to",
        sosActivated: "SOS activated! Emergency services notified."
    },
    hi: {
        ready: "सिस्टम तैयार है",
        cameraActive: "कैमरा सक्रिय है",
        start: "नेविगेशन शुरू हो रहा है",
        stop: "नेविगेशन बंद हो गया",
        listening: "सुन रहा हूँ...",
        processing: "प्रक्रिया जारी है...",
        speaking: "बोल रहा है...",
        textFound: "पाठ मिला",
        noText: "कोई पाठ नहीं मिला",
        navigating: "नेविगेट किया जा रहा है",
        sosActivated: "एसओएस सक्रिय! आपातकालीन सेवाएं सूचित की गई हैं।"
    },
    te: {
        ready: "సిస్టమ్ సిద్ధంగా ఉంది",
        cameraActive: "కెమెరా సక్రియంగా ఉంది",
        start: "నావిగేషన్ ప్రారంభమవుతోంది",
        stop: "నావిగేషన్ ఆపివేయబడింది",
        listening: "వినడం...",
        processing: "ప్రక్రియ చేస్తున్నాం...",
        speaking: "మాట్లాడుతోంది...",
        textFound: "టెక్స్ట్ కనుగొనబడింది",
        noText: "పాఠ్యం కనుగొనబడలేదు",
        navigating: "తరలిస్తున్నాము",
        sosActivated: "ఎస్‌ఓ‌ఎస్ సక్రియమైనది! ఎమర్జెన్సీ సేవలకు తెలియజేయబడింది."
    }
};

// =============== INITIALIZATION ===============
document.addEventListener('DOMContentLoaded', async () => {
    console.log('[INIT] Starting GuideX Front-end...');
    
    // Load language
    const savedLang = localStorage.getItem('guideXLanguage') || 'en';
    currentLanguage = savedLang;
    document.getElementById('languageSelect').value = currentLanguage;
    
    // Event listeners
    document.getElementById('languageSelect').addEventListener('change', (e) => {
        setLanguage(e.target.value);
    });

    // Initialize geolocation
    initializeGeolocation();
    
    // Load registered faces
    await loadRegisteredFaces();
    
    // Start live detections polling
    startDetectionPolling();
    
    // Setup speech recognition
    setupSpeechRecognition();
    
    // Check initial status
    try {
        const res = await fetch('/status');
        const data = await res.json();
        if (data.running) {
            isNavigating = true;
            document.getElementById('startNavBtn').disabled = true;
            document.getElementById('stopNavBtn').disabled = false;
            document.getElementById('cameraStatusText').textContent = translate('cameraActive');
            document.getElementById('videoFeed').src = '/video_feed';
            startAlertPolling();
        }
    } catch (e) {
        console.warn("Status check failed on load:", e);
    }
    
    showToast(translate('ready'));
});

// =============== LANGUAGE ===============
function translate(key) {
    // UI text must ALWAYS be in English!
    const langs = translations['en'];
    return langs[key] || key;
}

async function setLanguage(lang) {
    try {
        const response = await fetch('/set_language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: lang })
        });
        if (response.ok) {
            currentLanguage = lang;
            localStorage.setItem('guideXLanguage', lang);
            speak(currentLanguage === 'hi' ? 'भाषा बदलकर हिंदी कर दी गई है।' : 
                  currentLanguage === 'te' ? 'భాష తెలుగుకు మార్చబడింది.' : 
                  'Language changed to English.');
            
            await loadRegisteredFaces();
        }
    } catch (e) {
        console.error("Set language backend sync failed:", e);
    }
}

// =============== TEXT-TO-SPEECH ===============
let currentTtsAudio = null;

async function fetchTranslation(text, targetLang) {
    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, target_lang: targetLang })
        });
        const data = await response.json();
        return data.translated || text;
    } catch (e) {
        console.warn("Translation failed in speak():", e);
        return text;
    }
}

async function speak(text) {
    if (!text) return;
    
    // Stop any playing audio
    if (currentTtsAudio) {
        try {
            currentTtsAudio.pause();
        } catch (e) {}
        currentTtsAudio = null;
    }
    
    // Auto-translate if language is not English and input text is in English (lacks Devnagari or Telugu characters)
    if (currentLanguage === 'hi' && !/[\u0900-\u097f]/.test(text)) {
        text = await fetchTranslation(text, 'hi');
    } else if (currentLanguage === 'te' && !/[\u0c00-\u0c7f]/.test(text)) {
        text = await fetchTranslation(text, 'te');
    }
    
    // For English, use native Web Speech Synthesis (very high quality)
    if (currentLanguage === 'en') {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 1.0;
        speechSynthesis.cancel();
        speechSynthesis.speak(utterance);
        return;
    }
    
    // For Hindi and Telugu, use Google Translate TTS
    const lang = currentLanguage;
    const url = `https://translate.google.com/translate_tts?ie=UTF-8&tl=${lang}&client=tw-ob&q=${encodeURIComponent(text)}`;
    
    currentTtsAudio = new Audio(url);
    currentTtsAudio.play().catch(err => {
        console.warn("Google TTS audio play blocked, falling back to Web Speech Synthesis:", err);
        // Fallback to native synthesis
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang === 'hi' ? 'hi-IN' : 'te-IN';
        speechSynthesis.cancel();
        speechSynthesis.speak(utterance);
    });
}

// =============== SPEECH RECOGNITION ===============
function setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.log('[SPEECH] Speech Recognition not supported');
        return;
    }

    window.recognition = new SpeechRecognition();
    window.recognition.lang = currentLanguage === 'en' ? 'en-US' : 
                               currentLanguage === 'hi' ? 'hi-IN' : 'te-IN';
    window.recognition.continuous = false;
    window.recognition.interimResults = false;

    window.recognition.onstart = () => {
        recognitionActive = true;
        document.getElementById('voiceBtn').classList.add('listening');
        document.getElementById('voiceText').textContent = translate('listening');
        // Stop any active speech to avoid self-hearing loops
        speechSynthesis.cancel();
        if (currentTtsAudio) {
            try { currentTtsAudio.pause(); } catch(e) {}
            currentTtsAudio = null;
        }
    };

    window.recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        handleVoiceCommand(transcript.trim());
    };

    window.recognition.onend = () => {
        recognitionActive = false;
        document.getElementById('voiceBtn').classList.remove('listening');
        document.getElementById('voiceText').textContent = 'Click to Speak';
    };

    window.recognition.onerror = (event) => {
        console.log('[SPEECH] Error:', event.error);
        recognitionActive = false;
        document.getElementById('voiceBtn').classList.remove('listening');
        document.getElementById('voiceText').textContent = 'Click to Speak';
    };
}

function toggleVoice() {
    if (!window.recognition) {
        showToast('Speech recognition not available');
        return;
    }

    if (recognitionActive) {
        window.recognition.abort();
    } else {
        // Update language matches dynamic select dropdown
        window.recognition.lang = currentLanguage === 'en' ? 'en-US' : 
                                   currentLanguage === 'hi' ? 'hi-IN' : 'te-IN';
        window.recognition.start();
    }
}

// =============== VOICE COMMAND HANDLING ===============
function handleVoiceCommand(command) {
    console.log('[VOICE] Command:', command);
    const lower = command.toLowerCase();
    
    // Quick local checks
    if (lower.includes('start') || lower.includes('begin') || lower.includes('चालू करो') || lower.includes('ప్రారంభించు')) {
        startNavigation();
    } else if (lower.includes('stop') || lower.includes('बंद करो') || lower.includes('ఆపు')) {
        stopNavigation();
    } else if (lower.includes('who') && (lower.includes('front') || lower.includes('infront') || lower.includes('सामने') || lower.includes('ముందు'))) {
        recognizeFace();
    } else if (lower.includes('read') || lower.includes('text') || lower.includes('पढ़ो') || lower.includes('చదువు')) {
        readText();
    } else if (lower.includes('sos') || lower.includes('emergency') || lower.includes('help') || lower.includes('मदद') || lower.includes('సహాయం')) {
        triggerSOS();
    } else if (lower.startsWith('go to') || lower.startsWith('navigate to')) {
        const dest = command.replace(/go to|navigate to/gi, '').trim();
        if (dest) {
            document.getElementById('locationInput').value = dest;
            navigateTo();
        }
    } else {
        // Fallback: Ask assistant brain on the server
        askMickey(command);
    }
}

async function askMickey(messageText) {
    try {
        const payload = {
            message: messageText,
            lat: currentLocation ? currentLocation.lat : null,
            lng: currentLocation ? currentLocation.lng : null,
            activeDestination: document.getElementById('locationInput').value.trim()
        };
        const response = await fetch('/mickey', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (data.reply) {
            // Speak translated text
            speak(data.speech || data.reply);
            // Display English reply on UI
            showToast(data.reply);
            
            // Execute action if returned
            if (data.action) {
                executeMickeyAction(data);
            }
        }
    } catch (error) {
        console.error('[MICKEY] Assistant error:', error);
        speak('Mickey had trouble understanding that.');
    }
}

function executeMickeyAction(data) {
    const action = data.action;
    if (action === 'navigate') {
        const dest = data.destination;
        if (dest) {
            document.getElementById('locationInput').value = dest;
            const resultsDiv = document.getElementById('navResults');
            resultsDiv.innerHTML = `<div class="nav-result"><i class="fas fa-check"></i> Navigating to ${dest}</div>`;
            searchMapLocation(dest);
        }
    } else if (action === 'start') {
        startNavigation();
    } else if (action === 'stop') {
        stopNavigation();
    } else if (action === 'sos') {
        triggerSOS();
    } else if (action === 'read') {
        readText();
    } else if (action === 'identify_face') {
        recognizeFace();
    } else if (action === 'location') {
        speakMapLocation();
    } else if (action === 'clear_route') {
        clearRoute();
    }
}

// =============== NAVIGATION MODULE ===============
async function startNavigation() {
    if (isNavigating) return;
    
    try {
        const response = await fetch('/start', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'started') {
            isNavigating = true;
            document.getElementById('startNavBtn').disabled = true;
            document.getElementById('stopNavBtn').disabled = false;
            document.getElementById('cameraStatusText').textContent = translate('cameraActive');
            document.getElementById('videoFeed').src = '/video_feed';
            
            speak(translate('start'));
            showToast(translate('start'));
            startAlertPolling();
        }
    } catch (error) {
        console.error('[NAV] Error:', error);
        showToast('Navigation start failed');
    }
}

async function stopNavigation() {
    if (!isNavigating) return;
    
    try {
        const response = await fetch('/stop', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'stopped') {
            isNavigating = false;
            document.getElementById('startNavBtn').disabled = false;
            document.getElementById('stopNavBtn').disabled = true;
            document.getElementById('cameraStatusText').textContent = 'Inactive';
            document.getElementById('videoFeed').src = '#';
            
            speak(translate('stop'));
            showToast(translate('stop'));
            stopAlertPolling();
        }
    } catch (error) {
        console.error('[NAV] Error:', error);
        showToast('Navigation stop failed');
    }
}

// =============== FACE RECOGNITION MODULE ===============
async function registerFace() {
    const nameInput = document.getElementById('faceNameInput');
    const name = nameInput.value.trim();
    
    if (!name) {
        showToast('Please enter a name');
        speak('Please enter a name');
        return;
    }
    
    // Camera check portal activation
    if (!isNavigating) {
        speak(currentLanguage === 'hi' ? 'कैमरा शुरू किया जा रहा है। कृपया लेंस की ओर देखें।' : 
              currentLanguage === 'te' ? 'కెమెరాని ఆన్ చేస్తున్నాను. దయచేసి లెన్స్ వైపు చూడండి.' : 
              'Activating camera portal. Please face the camera lens.');
        await startNavigation();
        await new Promise(resolve => setTimeout(resolve, 2500));
    }
    
    try {
        const response = await fetch('/add_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            showToast(data.message); // Displays English message
            speak(data.speech || data.message); // Localized speech response from server
            nameInput.value = '';
            await loadRegisteredFaces();
        } else {
            showToast(data.error || 'Registration failed');
            speak(data.error || 'Registration failed');
        }
    } catch (error) {
        console.error('[FACE] Registration error:', error);
        showToast('Face registration failed');
    }
}

async function recognizeFace() {
    if (!isNavigating) {
        speak(currentLanguage === 'hi' ? 'कैमरा शुरू किया जा रहा है।' : 
              currentLanguage === 'te' ? 'కెమెరాని ఆన్ చేస్తున్నాను.' : 
              'Activating camera.');
        await startNavigation();
        await new Promise(resolve => setTimeout(resolve, 1500));
    }
    
    try {
        const response = await fetch('/identify_face', { method: 'POST' });
        const data = await response.json();
        
        const resultsDiv = document.getElementById('faceResults');
        
        if (data.name) {
            resultsDiv.innerHTML = `<div class="face-results">
                <div class="face-result">
                    <span class="face-name">${data.name}</span>
                    <span class="face-confidence">Matched</span>
                </div>
            </div>`;
            speak(data.speech || `${data.name} is in front of you`);
            showToast(`${data.name} is in front of you`);
        } else if (data.description) {
            resultsDiv.innerHTML = `<div class="face-results">
                <div class="face-result">
                    <span class="face-name">Unregistered</span>
                </div>
            </div>`;
            speak(data.speech || `The person in front of you is not registered.`);
            showToast(data.description);
        } else {
            resultsDiv.innerHTML = '<div class="no-result">No faces detected</div>';
            speak(data.speech || 'No one detected');
        }
    } catch (error) {
        console.error('[FACE] Recognition error:', error);
        showToast('Face recognition failed');
    }
}

async function loadRegisteredFaces() {
    try {
        const response = await fetch('/list_faces');
        const data = await response.json();
        
        registeredFaces = data.faces || [];
        
        // Update the faces count element
        const countEl = document.getElementById('registeredFacesCount');
        if (countEl) {
            countEl.textContent = registeredFaces.length;
        }
        
        const list = document.getElementById('registeredFacesList');
        
        if (registeredFaces.length === 0) {
            list.innerHTML = '<div class="no-result">No registered faces</div>';
        } else {
            let html = '<div class="faces-grid">';
            registeredFaces.forEach(name => {
                html += `<div class="face-item">
                    <span>👤 ${name}</span>
                    <button onclick="deleteFace('${name}')" class="btn-delete">×</button>
                </div>`;
            });
            html += '</div>';
            list.innerHTML = html;
        }
    } catch (error) {
        console.error('[FACE] Load error:', error);
    }
}

async function deleteFace(name) {
    const confirmMsg = currentLanguage === 'hi' ? `${name} को हटाना चाहते हैं?` :
                       currentLanguage === 'te' ? `${name}ని తొలగించాలా?` : `Are you sure you want to delete ${name}?`;
    if (!confirm(confirmMsg)) return;
    
    try {
        const response = await fetch('/delete_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await response.json();
        if (data.status === 'success') {
            showToast(data.message || `${name} deleted`);
            speak(data.speech || data.message || `${name} deleted`);
            await loadRegisteredFaces();
        } else {
            showToast(data.error || 'Delete failed');
        }
    } catch (error) {
        console.error('[FACE] Delete error:', error);
        showToast('Delete failed');
    }
}

// =============== TEXT RECOGNITION MODULE ===============
async function readText() {
    if (!isNavigating) {
        speak(currentLanguage === 'hi' ? 'स्कैन करने के लिए कृपया नेविगेशन चालू करें।' :
              currentLanguage === 'te' ? 'స్కాన్ చేయడానికి దయచేసి నావిగేషన్‌ను ప్రారంభించండి.' :
              'Please start navigation to read text.');
        showToast('Start navigation first');
        return;
    }
    
    try {
        const response = await fetch('/read_text');
        const data = await response.json();
        
        const resultsDiv = document.getElementById('textResults');
        
        if (data.text) {
            resultsDiv.innerHTML = `<div class="text-result">${data.text}</div>`;
            speak(data.speech || `Text found: ${data.text}`);
            showToast(translate('textFound'));
        } else {
            resultsDiv.innerHTML = '<div class="no-result">No text detected</div>';
            speak(translate('noText'));
        }
    } catch (error) {
        console.error('[OCR] Error:', error);
        showToast('Text recognition failed');
    }
}

// =============== DESTINATION NAVIGATION ===============
async function navigateTo() {
    const location = document.getElementById('locationInput').value.trim();
    if (!location) {
        showToast('Enter a location');
        return;
    }
    
    try {
        const response = await fetch('/navigate_to', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ location })
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            const resultsDiv = document.getElementById('navResults');
            resultsDiv.innerHTML = `<div class="nav-result">
                <i class="fas fa-check"></i> Navigating to ${location}
            </div>`;
            
            speak(data.message); // Server returns description in correct language
            showToast(`Searching route to ${location}...`);
            
            // Search and navigate on map
            searchMapLocation(location);
        } else {
            showToast(data.message || 'Route failed');
        }
    } catch (error) {
        console.error('[NAV] Error:', error);
        showToast('Navigation failed');
    }
}

// =============== EMERGENCY SOS ===============
function triggerSOS() {
    console.log('[SOS] Activated');
    speak(translate('sosActivated'));
    showToast(translate('sosActivated'));
    
    // Play siren sound locally
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 1000;
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 2.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 2.5);

    // Call backend emergency sos endpoint
    fetch('/sos', { method: 'POST' }).catch(err => console.error('[SOS] Backend call failed:', err));
}

// =============== DETECTION POLLING ===============
function startDetectionPolling() {
    setInterval(async () => {
        if (!isNavigating) return;
        
        try {
            const response = await fetch('/get_detections');
            const data = await response.json();
            
            // Update UI count cards
            const objCount = document.getElementById('detectionCount')?.querySelector('.value');
            if (objCount) objCount.textContent = data.detections?.length || 0;
            
            const faceCount = document.getElementById('faceCount')?.querySelector('.value');
            if (faceCount) faceCount.textContent = data.faces?.length || 0;
            
            const textCount = document.getElementById('textCount')?.querySelector('.value');
            if (textCount) textCount.textContent = data.text ? 'Yes' : 'None';
        } catch (error) {
            console.error('[POLLING] Detection polling error:', error);
        }
    }, 1500);
}

// =============== REALTIME ALERTS POLLING ===============
function startAlertPolling() {
    if (alertInterval) clearInterval(alertInterval);
    alertInterval = setInterval(async () => {
        if (!isNavigating) return;
        try {
            const response = await fetch('/get_alert');
            const data = await response.json();
            if (data.alert) {
                // Speak translated speech
                speak(data.speech || data.alert);
                // Show English warning toast
                showToast(data.alert);
            }
        } catch (error) {
            console.error('[ALERTS] Polling failed:', error);
        }
    }, 600);
}

function stopAlertPolling() {
    if (alertInterval) {
        clearInterval(alertInterval);
        alertInterval = null;
    }
}

// =============== GEOLOCATION MODULE ===============
function initializeGeolocation() {
    if (navigator.geolocation) {
        navigator.geolocation.watchPosition(
            (position) => {
                currentLocation = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude
                };
                
                // Expose to map logic
                updateMapLocation(currentLocation);
            },
            (error) => {
                console.warn('[GEO] Location watch error:', error);
            },
            { enableHighAccuracy: true, timeout: 10000 }
        );
    }
}

// =============== UI TOAST HELPER ===============
function showToast(message) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// =============== EXPOSE GLOBALS ===============
window.currentLocation = () => currentLocation;
window.translate = translate;
window.speak = speak;
window.showToast = showToast;