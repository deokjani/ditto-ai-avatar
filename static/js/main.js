/**
 * main.js - AI Avatar Chat Application
 * REST API for chat, WebSocket for video notifications
 */

// ===== 전역 변수 =====
let socket = null;          // WebSocket 인스턴스 (비디오 알림용)
let sessionId = null;       // 세션 ID
let isProcessing = false;   // 중복 전송 방지

/**
 * 세션 ID 생성 또는 가져오기
 */
function getSessionId() {
    if (!sessionId) {
        sessionId = 'session_' + Math.random().toString(36).substring(2) + Date.now().toString(36);
    }
    return sessionId;
}

// ===== 타이핑 효과 클래스 =====
class TypingEffect {
    constructor() {
        this.typingSpeed = 30;      // 글자당 30ms
        this.initialDelay = 3000;    // 시작 전 3초 대기
    }
    
    /**
     * 메시지를 타이핑 효과로 표시
     */
    async typeMessage(text, container, isUser = false) {
        // 사용자 메시지는 즉시 표시
        if (isUser) {
            this.addMessageInstant(text, container, true);
            return;
        }
        
        // AI 메시지 엘리먼트 생성
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
        
        // 타이핑 커서
        const typingCursor = document.createElement('span');
        typingCursor.className = 'typing-cursor';
        typingCursor.textContent = '▌';
        textDiv.appendChild(typingCursor);
        
        // 대기
        await this.delay(this.initialDelay);
        
        // 한 글자씩 타이핑
        for (let i = 0; i < text.length; i++) {
            textDiv.textContent = text.substring(0, i + 1);
            textDiv.appendChild(typingCursor);
            container.scrollTop = container.scrollHeight;
            await this.delay(this.typingSpeed);
        }
        
        // 커서 제거
        typingCursor.remove();
        messageDiv.classList.remove('typing-message');
        
        return messageDiv;
    }
    
    /**
     * 메시지를 즉시 표시
     */
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
    
    /**
     * 지연 함수
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * 현재 시간 포맷
     */
    getCurrentTime() {
        const now = new Date();
        const hours = now.getHours();
        const minutes = now.getMinutes().toString().padStart(2, '0');
        const period = hours >= 12 ? '오후' : '오전';
        const displayHours = hours > 12 ? hours - 12 : hours || 12;
        return `${period} ${displayHours}:${minutes}`;
    }
}

// ===== 비디오 매니저 클래스 =====
class VideoManager {
    constructor() {
        this.videoQueue = [];        // 재생 대기 큐
        this.currentLayerIndex = 0;  // 현재 활성 레이어
        this.nextLayerIndex = 1;     // 다음 레이어
        this.isTransitioning = false; // 전환 중 플래그
        this.currentVideoType = 'default';
        
        // 3개의 비디오 레이어 (트리플 버퍼링)
        this.layers = [
            { id: 1, element: document.getElementById('videoLayer1'), video: document.getElementById('video1') },
            { id: 2, element: document.getElementById('videoLayer2'), video: document.getElementById('video2') },
            { id: 3, element: document.getElementById('videoLayer3'), video: document.getElementById('video3') }
        ];
        
        this.audioPlayer = document.getElementById('audioPlayer');
        
        this.setupVideoEvents();
        this.initializeDefaultVideo();
    }
    
    /**
     * 기본 대기 비디오 초기화
     */
    initializeDefaultVideo() {
        const defaultVideo = this.layers[0].video;
        defaultVideo.src = '/default_video';
        defaultVideo.loop = true;
        defaultVideo.play().catch(e => {});
    }
    
    /**
     * 비디오 이벤트 설정
     */
    setupVideoEvents() {
        this.layers.forEach((layer, index) => {
            layer.video.addEventListener('ended', () => this.handleVideoEnded(index));
        });
    }
    
    /**
     * 현재 활성 레이어 반환
     */
    getCurrentLayer() {
        return this.layers[this.currentLayerIndex];
    }
    
    /**
     * 다음 레이어 반환
     */
    getNextLayer() {
        return this.layers[this.nextLayerIndex];
    }
    
    /**
     * 비디오 전환
     */
    async switchToVideo(url, videoType, audioUrl = null) {
        if (this.isTransitioning) return;
        
        this.isTransitioning = true;
        
        const currentLayer = this.getCurrentLayer();
        const nextLayer = this.getNextLayer();
        
        // 다음 레이어에 비디오 로드
        nextLayer.video.src = url;
        nextLayer.video.loop = (videoType === 'default');
        
        // 비디오 재생
        await nextLayer.video.play();
        
        // 오디오 재생
        if (audioUrl && videoType === 'response') {
            this.audioPlayer.src = audioUrl;
            this.audioPlayer.play();
        }
        
        // 페이드 전환
        nextLayer.element.classList.add('active');
        
        // 300ms 후 이전 레이어 숨기기
        setTimeout(() => {
            currentLayer.element.classList.remove('active');
            currentLayer.video.pause();
            
            // 레이어 인덱스 순환
            this.currentLayerIndex = this.nextLayerIndex;
            this.nextLayerIndex = (this.nextLayerIndex + 1) % 3;
            
            this.currentVideoType = videoType;
            this.isTransitioning = false;
        }, 300);
    }
    
    /**
     * 큐에서 다음 비디오 재생
     */
    async playNextVideo() {
        if (this.videoQueue.length === 0) return;
        
        const nextItem = this.videoQueue.shift();
        await this.switchToVideo(nextItem.url, 'response', nextItem.audioUrl);
        updateStatus('응답 중', 'processing');
    }
    
    /**
     * 비디오 종료 처리
     */
    async handleVideoEnded(layerIndex) {
        // 현재 활성 레이어만 처리
        if (layerIndex !== this.currentLayerIndex) return;
        
        if (this.videoQueue.length > 0) {
            // 다음 비디오 재생
            await this.playNextVideo();
        } else if (this.currentVideoType === 'response') {
            // 기본 비디오로 복귀
            await this.switchToVideo('/default_video', 'default', null);
            updateStatus('대기 중', 'waiting');
        }
    }
    
    /**
     * 비디오를 큐에 추가
     */
    addToQueue(videoData) {
        this.videoQueue.push(videoData);
        
        // 대기 중이면 즉시 재생
        if (this.currentVideoType === 'default' && !this.isTransitioning) {
            this.playNextVideo();
        }
    }
}

// ===== 인스턴스 생성 =====
const typingEffect = new TypingEffect();
const videoManager = new VideoManager();

// ===== WebSocket 연결 =====
/**
 * WebSocket 연결 초기화 (비디오 알림 전용)
 */
function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    // 연결 성공
    socket.onopen = function(event) {
        // 주기적으로 ping 전송 (연결 유지)
        setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send('ping');
            }
        }, 30000);  // 30초마다
    };
    
    // 연결 종료
    socket.onclose = function(event) {
        setTimeout(connectWebSocket, 3000);  // 3초 후 재연결
    };
    
    // 메시지 수신
    socket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleVideoMessage(data);
        } catch (e) {
            // ping/pong 메시지는 무시
        }
    };
}

/**
 * 비디오 메시지 처리
 */
function handleVideoMessage(data) {
    if (data.type === 'video_ready') {
        // 비디오 생성 완료 - 큐에 추가
        videoManager.addToQueue({
            url: data.data.video_url,
            audioUrl: data.data.audio_url,
            text: data.data.text
        });
    }
}

// ===== UI 함수 =====
/**
 * 상태 표시 업데이트
 */
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

/**
 * 메시지 전송 (REST API 사용)
 */
window.sendMessage = async function() {
    const input = document.getElementById('messageInput');
    const text = input.value.trim();
    
    // 빈 메시지 체크
    if (!text) return;
    
    // 중복 전송 방지
    if (isProcessing) return;
    
    isProcessing = true;
    
    // 사용자 메시지 표시
    const messagesContainer = document.getElementById('chatMessages');
    typingEffect.addMessageInstant(text, messagesContainer, true);
    
    // 입력 필드 초기화
    input.value = '';
    
    // 상태 업데이트
    updateStatus('생각 중...', 'processing');
    
    try {
        // REST API로 채팅 요청
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
        
        // AI 응답 표시 (타이핑 효과)
        await typingEffect.typeMessage(data.response, messagesContainer, false);
        
        // 상태 업데이트
        updateStatus('생각 중...', 'processing');
        
    } catch (error) {
        updateStatus('오류 발생', '');
        
        // 오류 메시지 표시
        typingEffect.addMessageInstant(
            '죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.',
            messagesContainer,
            false
        );
    } finally {
        isProcessing = false;
    }
}

/**
 * 엔터 키 처리
 */
window.handleKeyPress = function(event) {
    if (event.key === 'Enter' && !isProcessing) {
        window.sendMessage();
    }
}

// ===== 초기화 =====
/**
 * 페이지 로드 시 초기화
 */
window.addEventListener('load', function() {
    // WebSocket 연결 (비디오 알림용)
    connectWebSocket();
    
    // 비디오 자동 재생 시도
    const initPlay = () => {
        videoManager.getCurrentLayer().video.play().catch(e => {});
    };
    
    initPlay();
    
    // 첫 클릭 시 재생 (자동 재생 실패 대비)
    document.addEventListener('click', function firstClick() {
        initPlay();
    }, { once: true });
    
    // 초기 상태 설정
    updateStatus('대기 중', 'waiting');
});
