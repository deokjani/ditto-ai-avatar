/**
 * main.js - AI Avatar Chat
 */

// ===== 전역 변수 =====
let socket = null;
let sessionId = null;
let isProcessing = false;

/**
 * 세션 ID 생성
 */
function getSessionId() {
    if (!sessionId) {
        sessionId = 'session_' + Math.random().toString(36).substring(2) + Date.now().toString(36);
    }
    return sessionId;
}

// ===== 타이핑 효과 =====
class TypingEffect {
    constructor() {
        this.typingSpeed = 30;
        this.initialDelay = 3000;
    }
    
    async typeMessage(text, container, isUser = false) {
        if (isUser) {
            this.addMessageInstant(text, container, true);
            return;
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai typing-message';
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = this.getCurrentTime();
        
        messageDiv.appendChild(textDiv);
        messageDiv.appendChild(timeDiv);
        container.appendChild(messageDiv);
        
        const typingCursor = document.createElement('span');
        typingCursor.className = 'typing-cursor';
        typingCursor.textContent = '▌';
        textDiv.appendChild(typingCursor);
        
        await this.delay(this.initialDelay);
        
        for (let i = 0; i < text.length; i++) {
            textDiv.textContent = text.substring(0, i + 1);
            textDiv.appendChild(typingCursor);
            container.scrollTop = container.scrollHeight;
            await this.delay(this.typingSpeed);
        }
        
        typingCursor.remove();
        messageDiv.classList.remove('typing-message');
        
        return messageDiv;
    }
    
    addMessageInstant(text, container, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.textContent = text;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = this.getCurrentTime();
        
        messageDiv.appendChild(textDiv);
        messageDiv.appendChild(timeDiv);
        
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
        
        return messageDiv;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    getCurrentTime() {
        const now = new Date();
        const hours = now.getHours();
        const minutes = now.getMinutes().toString().padStart(2, '0');
        const period = hours >= 12 ? '오후' : '오전';
        const displayHours = hours > 12 ? hours - 12 : hours || 12;
        return `${period} ${displayHours}:${minutes}`;
    }
}

// ===== 비디오 매니저 =====
class VideoManager {
    constructor() {
        this.videoQueue = [];
        this.currentLayerIndex = 0;
        this.nextLayerIndex = 1;
        this.isTransitioning = false;
        this.currentVideoType = 'default';
        
        this.layers = [
            { id: 1, element: document.getElementById('videoLayer1'), video: document.getElementById('video1') },
            { id: 2, element: document.getElementById('videoLayer2'), video: document.getElementById('video2') },
            { id: 3, element: document.getElementById('videoLayer3'), video: document.getElementById('video3') }
        ];
        
        this.audioPlayer = document.getElementById('audioPlayer');
        
        this.setupVideoEvents();
        this.initializeDefaultVideo();
    }
    
    initializeDefaultVideo() {
        const defaultVideo = this.layers[0].video;
        const defaultLayer = this.layers[0].element;
        defaultVideo.src = '/default_video';
        defaultVideo.loop = true;
        defaultVideo.play().catch(e => {});
        defaultLayer.classList.add('active');
    }
    
    setupVideoEvents() {
        this.layers.forEach((layer, index) => {
            layer.video.addEventListener('ended', () => this.handleVideoEnded(index));
        });
    }
    
    getCurrentLayer() {
        return this.layers[this.currentLayerIndex];
    }
    
    getNextLayer() {
        return this.layers[this.nextLayerIndex];
    }
    
    async switchToVideo(url, videoType, audioUrl = null) {
        if (this.isTransitioning) return;
        
        this.isTransitioning = true;
        
        const currentLayer = this.getCurrentLayer();
        const nextLayer = this.getNextLayer();
        
        nextLayer.video.src = url;
        nextLayer.video.loop = (videoType === 'default');
        
        await nextLayer.video.play();
        
        if (audioUrl && videoType === 'response') {
            this.audioPlayer.src = audioUrl;
            this.audioPlayer.play();
        }
        
        nextLayer.element.classList.add('active');
        
        setTimeout(() => {
            currentLayer.element.classList.remove('active');
            currentLayer.video.pause();

            if (videoType !== 'default') {
                currentLayer.video.src = '';
            }
            
            this.currentLayerIndex = this.nextLayerIndex;
            this.nextLayerIndex = (this.nextLayerIndex + 1) % 3;
            
            this.currentVideoType = videoType;
            this.isTransitioning = false;
        }, 300);
    }
    
    async playNextVideo() {
        if (this.videoQueue.length === 0) return;
        
        const nextItem = this.videoQueue.shift();
        await this.switchToVideo(nextItem.url, 'response', nextItem.audioUrl);
        updateStatus('응답 중', 'processing');
    }
    
    async handleVideoEnded(layerIndex) {
        if (layerIndex !== this.currentLayerIndex) return;
        
        if (this.videoQueue.length > 0) {
            await this.playNextVideo();
        } else if (this.currentVideoType === 'response') {
            await this.switchToVideo('/default_video', 'default', null);
            updateStatus('대기 중', 'waiting');
        }
    }
    
    addToQueue(videoData) {
        this.videoQueue.push(videoData);
        
        if (this.currentVideoType === 'default' && !this.isTransitioning) {
            this.playNextVideo();
        }
    }
}

// ===== 인스턴스 생성 =====
const typingEffect = new TypingEffect();
const videoManager = new VideoManager();

// ===== WebSocket 연결 =====
function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = function(event) {
        setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send('ping');
            }
        }, 30000);
    };
    
    socket.onclose = function(event) {
        setTimeout(connectWebSocket, 3000);
    };
    
    socket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleVideoMessage(data);
        } catch (e) {}
    };
}

function handleVideoMessage(data) {
    if (data.type === 'video_ready') {
        videoManager.addToQueue({
            url: data.data.video_url,
            audioUrl: data.data.audio_url,
            text: data.data.text
        });
    }
}

// ===== UI 함수 =====
function updateStatus(text, type) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    statusText.textContent = text;
    statusDot.className = 'status-dot';
    
    if (type === 'waiting') {
        statusDot.classList.add('waiting');
    } else if (type === 'processing') {
        statusDot.classList.add('processing');
    }
}

window.sendMessage = async function() {
    const input = document.getElementById('messageInput');
    const text = input.value.trim();
    
    if (!text || isProcessing) return;
    
    isProcessing = true;
    
    const messagesContainer = document.getElementById('chatMessages');
    typingEffect.addMessageInstant(text, messagesContainer, true);
    
    input.value = '';
    updateStatus('생각 중...', 'processing');
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                session_id: getSessionId()
            })
        });
        
        if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`);
        }
        
        const data = await response.json();
        
        await typingEffect.typeMessage(data.response, messagesContainer, false);
        updateStatus('생각 중...', 'processing');
        
    } catch (error) {
        updateStatus('오류 발생', '');
        typingEffect.addMessageInstant(
            '죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.',
            messagesContainer,
            false
        );
    } finally {
        isProcessing = false;
    }
}

window.handleKeyPress = function(event) {
    if (event.key === 'Enter' && !isProcessing) {
        window.sendMessage();
    }
}

// ===== 초기화 =====
window.addEventListener('load', function() {
    connectWebSocket();
    
    const initPlay = () => {
        videoManager.getCurrentLayer().video.play().catch(e => {});
    };
    
    initPlay();
    
    document.addEventListener('click', function firstClick() {
        initPlay();
    }, { once: true });
    
    updateStatus('대기 중', 'waiting');
});
