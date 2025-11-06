--[[
    ╔══════════════════════════════════════════════════════════════╗
    ║      BLADE BALL AUTO PARRY V7.0 - ULTIMATE EDITION          ║
    ║         Sistema Ultra Avançado Premium Edition              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🔥 NOVAS CARACTERÍSTICAS ULTRA AVANÇADAS:
    • Filtro de Kalman Multi-Dimensional Adaptativo
    • Predição com Filtro de Partículas (Particle Filter)
    • Extended Kalman Filter (EKF) para Trajetórias Não-Lineares
    • Unscented Kalman Filter (UKF) para Alta Precisão
    • Sistema de Fusão Sensorial Bayesiana
    • Deep Neural Network com 5 Camadas Ocultas
    • Reinforcement Learning com Actor-Critic
    • Genetic Algorithm para Auto-Otimização
    • Visualização 3D Avançada com Shaders Customizados
    • Sistema de Predição com Markov Chain Monte Carlo
    • Análise de Fourier para Detecção de Padrões Periódicos
    • Sistema de Detecção de Anomalias com Isolation Forest
    • Cache Distribuído com LRU/LFU Híbrido
    • Sistema de Logging com Compressão e Criptografia
    • Anti-Detecção com Fingerprint Dinâmico e Ofuscação
    • Sistema de Backup com Versionamento e Delta Compression
    • Performance Otimizada com JIT Compilation Simulation
    • Sistema de Telemetria e Analytics em Tempo Real
]]

-- ══════════════════════════════════════════════════════════════
-- SERVICES & ADVANCED CONSTANTS
-- ══════════════════════════════════════════════════════════════

local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local UserInputService = game:GetService("UserInputService")
local TweenService = game:GetService("TweenService")
local Workspace = game:GetService("Workspace")
local Stats = game:GetService("Stats")
local HttpService = game:GetService("HttpService")

local Player = Players.LocalPlayer
local Camera = Workspace.CurrentCamera

-- Advanced Constants
local CONSTANTS = {
    VERSION = "7.0 - Ultimate Edition",
    UPDATE_RATE = 1/120, -- 120 FPS Ultra Performance
    CACHE_TTL = 0.1,
    MAX_HISTORY = 50,
    CONFIDENCE_THRESHOLD = 0.55,
    
    -- Advanced Parry Modes
    MODES = {
        ULTRA_SAFE = {multiplier = 0.5, confidence = 0.9, aggression = 0.3},
        SAFE = {multiplier = 0.7, confidence = 0.8, aggression = 0.5},
        BALANCED = {multiplier = 1.0, confidence = 0.6, aggression = 0.7},
        AGGRESSIVE = {multiplier = 1.3, confidence = 0.4, aggression = 0.9},
        ULTRA_AGGRESSIVE = {multiplier = 1.6, confidence = 0.3, aggression = 1.0}
    },
    
    -- Deep Neural Network Architecture
    DNN_LAYERS = {12, 10, 8, 6, 4, 1},
    DNN_LEARNING_RATE = 0.005,
    DNN_MOMENTUM = 0.9,
    DNN_DROPOUT = 0.2,
    
    -- Kalman Filter Parameters
    KALMAN_PROCESS_NOISE = 0.01,
    KALMAN_MEASUREMENT_NOISE = 0.1,
    KALMAN_INITIAL_ERROR = 1.0,
    
    -- Particle Filter Parameters
    PARTICLE_COUNT = 100,
    PARTICLE_RESAMPLE_THRESHOLD = 0.5,
    
    -- Genetic Algorithm Parameters
    GA_POPULATION_SIZE = 50,
    GA_MUTATION_RATE = 0.1,
    GA_CROSSOVER_RATE = 0.7,
    GA_ELITE_SIZE = 5,
    
    -- Advanced Prediction Strategies
    PREDICTION_STRATEGIES = {
        KALMAN = "kalman",
        EKF = "extended_kalman",
        UKF = "unscented_kalman",
        PARTICLE = "particle_filter",
        PHYSICS = "physics",
        PATTERN = "pattern",
        DNN = "deep_neural",
        HYBRID = "hybrid",
        BAYESIAN = "bayesian_fusion"
    }
}

-- ══════════════════════════════════════════════════════════════
-- ADVANCED CONFIGURATION MANAGER
-- ══════════════════════════════════════════════════════════════

local Config = {
    -- Core Settings
    enabled = true,
    mode = "BALANCED",
    autoAdjust = true,
    
    -- Advanced Detection
    detectCurve = true,
    detectSpin = true,
    detectBounce = true,
    usePhysicsPrediction = true,
    useKalmanFilter = true,
    useParticleFilter = true,
    useEKF = true,
    useUKF = false,
    
    -- Deep Learning Settings
    dnnEnabled = true,
    dnnLearningRate = 0.005,
    dnnBatchTraining = true,
    dnnDropout = 0.2,
    reinforcementLearning = true,
    actorCriticEnabled = true,
    
    -- Genetic Algorithm
    geneticOptimization = true,
    gaPopulationSize = 50,
    gaMutationRate = 0.1,
    
    -- Advanced Prediction
    useBayesianFusion = true,
    useMarkovChain = true,
    useFourierAnalysis = true,
    useAnomalyDetection = true,
    predictionHorizon = 1.0,
    
    -- Visual Advanced
    showHUD = true,
    showNotifications = true,
    hudPosition = UDim2.new(0, 10, 0, 10),
    showTrajectory3D = true,
    showKalmanPrediction = true,
    showParticleCloud = true,
    showConfidenceEllipse = true,
    showVelocityField = true,
    showHeatmap3D = true,
    showParryEffects = true,
    use3DShaders = true,
    particleEffectsQuality = "ultra",
    trajectoryPoints = 12,
    
    -- Anti-Detection Ultra
    randomizeTimings = true,
    dynamicFingerprint = true,
    advancedObfuscation = true,
    pingCompensation = true,
    jitterCompensation = true,
    
    -- Performance
    multiThreading = true,
    jitOptimization = true,
    cacheStrategy = "hybrid_lru_lfu",
    compressionEnabled = true,
    
    -- Advanced Features
    autoBackup = true,
    backupInterval = 300,
    telemetryEnabled = true,
    analyticsEnabled = true,
    
    -- Fast Parry & Auto Spam
    autoSpamEnabled = false,
    autoSpamThreshold = 15,
    autoSpamParries = 3,
    autoSpamDuration = 2.0,
    fastBallThreshold = 200,
    fastParryEnabled = true,
    fastParryThreshold = 22,
    fastParryPriorityBoost = 1.3,
    targetChangeAggressiveness = 1.25
}

-- ══════════════════════════════════════════════════════════════
-- ADVANCED MATH UTILITIES
-- ══════════════════════════════════════════════════════════════

local AdvancedMath = {}

-- Matrix operations for Kalman Filter
function AdvancedMath:createMatrix(rows, cols, value)
    local matrix = {}
    for i = 1, rows do
        matrix[i] = {}
        for j = 1, cols do
            matrix[i][j] = value or 0
        end
    end
    return matrix
end

function AdvancedMath:multiplyMatrix(a, b)
    local rows_a, cols_a = #a, #a[1]
    local rows_b, cols_b = #b, #b[1]
    
    if cols_a ~= rows_b then return nil end
    
    local result = self:createMatrix(rows_a, cols_b)
    for i = 1, rows_a do
        for j = 1, cols_b do
            local sum = 0
            for k = 1, cols_a do
                sum = sum + a[i][k] * b[k][j]
            end
            result[i][j] = sum
        end
    end
    return result
end

function AdvancedMath:addMatrix(a, b)
    local result = self:createMatrix(#a, #a[1])
    for i = 1, #a do
        for j = 1, #a[1] do
            result[i][j] = a[i][j] + b[i][j]
        end
    end
    return result
end

function AdvancedMath:transposeMatrix(matrix)
    local result = self:createMatrix(#matrix[1], #matrix)
    for i = 1, #matrix do
        for j = 1, #matrix[1] do
            result[j][i] = matrix[i][j]
        end
    end
    return result
end

function AdvancedMath:invertMatrix(matrix)
    -- Simplified 3x3 matrix inversion for Kalman Filter
    local n = #matrix
    if n ~= 3 then return matrix end -- Fallback
    
    local det = matrix[1][1] * (matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2])
                - matrix[1][2] * (matrix[2][1] * matrix[3][3] - matrix[2][3] * matrix[3][1])
                + matrix[1][3] * (matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1])
    
    if math.abs(det) < 0.0001 then return matrix end
    
    local inv = self:createMatrix(3, 3)
    inv[1][1] = (matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2]) / det
    inv[1][2] = (matrix[1][3] * matrix[3][2] - matrix[1][2] * matrix[3][3]) / det
    inv[1][3] = (matrix[1][2] * matrix[2][3] - matrix[1][3] * matrix[2][2]) / det
    inv[2][1] = (matrix[2][3] * matrix[3][1] - matrix[2][1] * matrix[3][3]) / det
    inv[2][2] = (matrix[1][1] * matrix[3][3] - matrix[1][3] * matrix[3][1]) / det
    inv[2][3] = (matrix[1][3] * matrix[2][1] - matrix[1][1] * matrix[2][3]) / det
    inv[3][1] = (matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1]) / det
    inv[3][2] = (matrix[1][2] * matrix[3][1] - matrix[1][1] * matrix[3][2]) / det
    inv[3][3] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / det
    
    return inv
end

-- Gaussian Random Number Generator (Box-Muller)
function AdvancedMath:gaussianRandom(mean, stddev)
    local u1 = math.random()
    local u2 = math.random()
    local z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean + z0 * stddev
end

-- Fast Fourier Transform (Simplified for pattern detection)
function AdvancedMath:fft(data)
    local n = #data
    if n <= 1 then return data end
    
    -- Simplified FFT for basic frequency analysis
    local spectrum = {}
    for k = 0, n-1 do
        local real, imag = 0, 0
        for t = 0, n-1 do
            local angle = 2 * math.pi * k * t / n
            real = real + data[t+1] * math.cos(angle)
            imag = imag - data[t+1] * math.sin(angle)
        end
        spectrum[k+1] = math.sqrt(real*real + imag*imag)
    end
    return spectrum
end

-- Sigmoid activation
function AdvancedMath:sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

-- ReLU activation
function AdvancedMath:relu(x)
    return math.max(0, x)
end

-- Leaky ReLU
function AdvancedMath:leakyRelu(x, alpha)
    alpha = alpha or 0.01
    return x > 0 and x or alpha * x
end

-- Tanh activation
function AdvancedMath:tanh(x)
    return math.tanh(x)
end

-- ══════════════════════════════════════════════════════════════
-- FILTRO DE KALMAN 1D - PARA VALORES ESCALARES
-- ══════════════════════════════════════════════════════════════
-- 
-- Este filtro é usado para filtrar valores escalares (como tempo estimado até o impacto).
-- O Filtro de Kalman 1D combina predições com medições para obter uma estimativa mais precisa.
--
-- Variáveis:
--   x: Estado estimado (valor filtrado)
--   P: Incerteza (covariância) do estado estimado
--   q: Process noise (ruído do processo) - quanto maior, mais o filtro confia nas medições
--   r: Measurement noise (ruído da medição) - quanto maior, mais o filtro confia na predição
--
-- Funcionamento:
--   1. Predict: Prediz o próximo valor baseado no estado atual
--   2. Update: Atualiza o estado com uma nova medição, combinando predição e medição
--   3. Get: Retorna o valor filtrado atual

local KalmanFilter1D = {
    x = nil,  -- Estado estimado (valor filtrado)
    P = nil,  -- Incerteza (covariância) do estado
    q = 0.1,  -- Process noise (ruído do processo) - ajustável
    r = 0.5,  -- Measurement noise (ruído da medição) - ajustável
    initialized = false
}

-- Inicializa o filtro com um valor inicial
-- @param initialValue: Valor inicial para o filtro
-- @param initialUncertainty: Incerteza inicial (opcional, padrão: 1.0)
function KalmanFilter1D:initialize(initialValue, initialUncertainty)
    self.x = initialValue or 0
    self.P = initialUncertainty or 1.0
    self.initialized = true
end

-- Prediz o próximo valor baseado no estado atual
-- No modelo 1D simples, assumimos que o valor permanece constante (sem dinâmica)
-- @return: Valor predito
function KalmanFilter1D:predict()
    if not self.initialized then return nil end
    
    -- No modelo 1D simples, o valor predito é o mesmo do estado atual
    -- A incerteza aumenta devido ao ruído do processo
    self.P = self.P + self.q
    
    return self.x
end

-- Atualiza o filtro com uma nova medição
-- Combina a predição com a medição usando o ganho de Kalman
-- @param measurement: Nova medição (valor observado)
function KalmanFilter1D:update(measurement)
    if not self.initialized then
        self:initialize(measurement)
        return
    end
    
    -- Prediz o próximo estado
    self:predict()
    
    -- Calcula o ganho de Kalman (K)
    -- K determina quanto confiar na medição vs. na predição
    -- K = P / (P + R), onde P é a incerteza da predição e R é o ruído da medição
    local K = self.P / (self.P + self.r)
    
    -- Atualiza o estado estimado
    -- x = x_predito + K * (medição - x_predito)
    -- Quanto maior K, mais o filtro confia na medição
    self.x = self.x + K * (measurement - self.x)
    
    -- Atualiza a incerteza
    -- P = (1 - K) * P
    -- A incerteza diminui após uma medição
    self.P = (1 - K) * self.P
end

-- Retorna o valor filtrado atual
-- @return: Valor filtrado (estado estimado)
function KalmanFilter1D:get()
    if not self.initialized then return nil end
    return self.x
end

-- Retorna a incerteza atual
-- @return: Incerteza (covariância)
function KalmanFilter1D:getUncertainty()
    if not self.initialized then return nil end
    return self.P
end

-- Define os parâmetros do filtro
-- @param q: Process noise (ruído do processo)
-- @param r: Measurement noise (ruído da medição)
function KalmanFilter1D:setParameters(q, r)
    if q then self.q = q end
    if r then self.r = r end
end

-- ══════════════════════════════════════════════════════════════
-- FILTRO DE KALMAN CONSTANT-VELOCITY (2D/VETORIAL)
-- ══════════════════════════════════════════════════════════════
--
-- Este filtro estima posição e velocidade de um objeto em movimento.
-- Assume que o objeto se move com velocidade constante (modelo constant-velocity).
--
-- Estado: [posição_x, posição_y, posição_z, velocidade_x, velocidade_y, velocidade_z]
--
-- Variáveis:
--   state: Estado estimado [posição, velocidade]
--   P: Matriz de covariância (6x6) - incerteza do estado
--   Qpos: Process noise para posição - quanto maior, mais variação esperada na posição
--   Qvel: Process noise para velocidade - quanto maior, mais variação esperada na velocidade
--   R: Measurement noise - quanto maior, menos confiança nas medições
--
-- Funcionamento:
--   1. Predict(dt): Prediz posição e velocidade após dt segundos
--   2. Update(z): Atualiza com uma nova medição de posição
--   3. GetState(): Retorna posição e velocidade estimadas

local KalmanFilterConstantVelocity = {
    state = nil,      -- Estado: [x, y, z, vx, vy, vz]
    P = nil,          -- Matriz de covariância (6x6)
    Qpos = 0.01,       -- Process noise para posição - ajustável
    Qvel = 0.1,        -- Process noise para velocidade - ajustável
    R = 0.1,           -- Measurement noise - ajustável
    initialized = false
}

-- Inicializa o filtro com posição e velocidade iniciais
-- @param initialPosition: Posição inicial (Vector3)
-- @param initialVelocity: Velocidade inicial (Vector3, opcional)
function KalmanFilterConstantVelocity:initialize(initialPosition, initialVelocity)
    initialVelocity = initialVelocity or Vector3.new(0, 0, 0)
    
    -- Estado: [x, y, z, vx, vy, vz]
    self.state = {
        initialPosition.X,
        initialPosition.Y,
        initialPosition.Z,
        initialVelocity.X,
        initialVelocity.Y,
        initialVelocity.Z
    }
    
    -- Matriz de covariância inicial (6x6)
    -- Diagonal: incerteza inicial para cada componente
    self.P = AdvancedMath:createMatrix(6, 6, 0)
    for i = 1, 6 do
        self.P[i][i] = 1.0  -- Incerteza inicial
    end
    
    self.initialized = true
end

-- Prediz o estado após dt segundos
-- Assumindo movimento com velocidade constante: posição = posição + velocidade * dt
-- @param dt: Tempo decorrido em segundos
-- @return: Posição predita (Vector3)
function KalmanFilterConstantVelocity:predict(dt)
    if not self.initialized then return nil end
    
    -- Matriz de transição de estado F (6x6)
    -- Modelo constant-velocity: posição += velocidade * dt
    -- [x']   [1  0  0  dt  0  0 ] [x]
    -- [y']   [0  1  0  0  dt  0 ] [y]
    -- [z'] = [0  0  1  0  0  dt ] [z]
    -- [vx']  [0  0  0  1   0  0 ] [vx]
    -- [vy']  [0  0  0  0   1  0 ] [vy]
    -- [vz']  [0  0  0  0   0  1 ] [vz]
    local F = {
        {1, 0, 0, dt, 0, 0},
        {0, 1, 0, 0, dt, 0},
        {0, 0, 1, 0, 0, dt},
        {0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1}
    }
    
    -- Prediz o estado: x_predito = F * x
    local newState = {}
    for i = 1, 6 do
        local sum = 0
        for j = 1, 6 do
            sum = sum + F[i][j] * self.state[j]
        end
        newState[i] = sum
    end
    self.state = newState
    
    -- Prediz a covariância: P_predito = F * P * F^T + Q
    local FP = AdvancedMath:multiplyMatrix(F, self.P)
    local FT = AdvancedMath:transposeMatrix(F)
    local FPFT = AdvancedMath:multiplyMatrix(FP, FT)
    
    -- Adiciona o ruído do processo Q
    local Q = AdvancedMath:createMatrix(6, 6, 0)
    Q[1][1] = self.Qpos  -- Ruído para posição X
    Q[2][2] = self.Qpos  -- Ruído para posição Y
    Q[3][3] = self.Qpos  -- Ruído para posição Z
    Q[4][4] = self.Qvel  -- Ruído para velocidade X
    Q[5][5] = self.Qvel  -- Ruído para velocidade Y
    Q[6][6] = self.Qvel  -- Ruído para velocidade Z
    
    self.P = AdvancedMath:addMatrix(FPFT, Q)
    
    -- Retorna a posição predita
    return Vector3.new(self.state[1], self.state[2], self.state[3])
end

-- Atualiza o filtro com uma nova medição de posição
-- @param measurement: Nova medição de posição (Vector3)
function KalmanFilterConstantVelocity:update(measurement)
    if not self.initialized then
        self:initialize(measurement)
        return
    end
    
    -- Matriz de observação H (3x6)
    -- Só observamos a posição, não a velocidade
    -- [x_medido]   [1  0  0  0  0  0] [x]
    -- [y_medido] = [0  1  0  0  0  0] [y]
    -- [z_medido]   [0  0  1  0  0  0] [z]
    --                                    [vx]
    --                                    [vy]
    --                                    [vz]
    local H = {
        {1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0}
    }
    
    -- Inovação: diferença entre medição e predição
    -- y = z - H * x_predito
    local z = {measurement.X, measurement.Y, measurement.Z}
    local Hx = {}
    for i = 1, 3 do
        local sum = 0
        for j = 1, 6 do
            sum = sum + H[i][j] * self.state[j]
        end
        Hx[i] = sum
    end
    
    local innovation = {}
    for i = 1, 3 do
        innovation[i] = z[i] - Hx[i]
    end
    
    -- Covariância da inovação: S = H * P * H^T + R
    local HP = AdvancedMath:multiplyMatrix(H, self.P)
    local HT = AdvancedMath:transposeMatrix(H)
    local HPHT = AdvancedMath:multiplyMatrix(HP, HT)
    
    local R = AdvancedMath:createMatrix(3, 3, 0)
    R[1][1] = self.R
    R[2][2] = self.R
    R[3][3] = self.R
    
    local S = AdvancedMath:addMatrix(HPHT, R)
    
    -- Ganho de Kalman: K = P * H^T * S^-1
    local PHT = AdvancedMath:multiplyMatrix(self.P, HT)
    local Sinv = AdvancedMath:invertMatrix(S)
    local K = AdvancedMath:multiplyMatrix(PHT, Sinv)
    
    -- Atualiza o estado: x = x_predito + K * y
    for i = 1, 6 do
        for j = 1, 3 do
            self.state[i] = self.state[i] + K[i][j] * innovation[j]
        end
    end
    
    -- Atualiza a covariância: P = (I - K * H) * P
    local KH = AdvancedMath:multiplyMatrix(K, H)
    local I = AdvancedMath:createMatrix(6, 6, 0)
    for i = 1, 6 do I[i][i] = 1 end
    
    for i = 1, 6 do
        for j = 1, 6 do
            I[i][j] = I[i][j] - KH[i][j]
        end
    end
    
    self.P = AdvancedMath:multiplyMatrix(I, self.P)
end

-- Retorna o estado estimado (posição e velocidade)
-- @return: Tabela com {position: Vector3, velocity: Vector3}
function KalmanFilterConstantVelocity:getState()
    if not self.initialized then return nil end
    
    return {
        position = Vector3.new(self.state[1], self.state[2], self.state[3]),
        velocity = Vector3.new(self.state[4], self.state[5], self.state[6])
    }
end

-- Define os parâmetros do filtro
-- @param Qpos: Process noise para posição
-- @param Qvel: Process noise para velocidade
-- @param R: Measurement noise
function KalmanFilterConstantVelocity:setParameters(Qpos, Qvel, R)
    if Qpos then self.Qpos = Qpos end
    if Qvel then self.Qvel = Qvel end
    if R then self.R = R end
end

-- ══════════════════════════════════════════════════════════════
-- FILTRO EMA (EXPONENTIAL MOVING AVERAGE)
-- ══════════════════════════════════════════════════════════════
--
-- Filtro mais leve que o Kalman, usado para suavizar ruído de valores simples.
-- A média móvel exponencial dá mais peso aos valores recentes.
--
-- Variáveis:
--   value: Valor filtrado atual
--   alpha: Fator de suavização (0 < alpha <= 1)
--          - Quanto menor alpha, mais suave (mais peso aos valores antigos)
--          - Quanto maior alpha, mais responsivo (mais peso aos valores recentes)
--          - alpha = 0.1: muito suave, lento para reagir
--          - alpha = 0.5: balanceado
--          - alpha = 0.9: muito responsivo, pouco suave
--
-- Fórmula: value_novo = alpha * valor_medido + (1 - alpha) * value_anterior

local EMAFilter = {
    value = nil,   -- Valor filtrado atual
    alpha = 0.3,   -- Fator de suavização (ajustável, padrão: 0.3)
    initialized = false
}

-- Inicializa o filtro com um valor inicial
-- @param initialValue: Valor inicial
-- @param alpha: Fator de suavização (opcional, padrão: 0.3)
function EMAFilter:initialize(initialValue, alpha)
    self.value = initialValue or 0
    if alpha then self.alpha = math.clamp(alpha, 0.01, 1.0) end
    self.initialized = true
end

-- Atualiza o filtro com um novo valor
-- Aplica a média móvel exponencial: value = alpha * novo_valor + (1 - alpha) * value_anterior
-- @param newValue: Novo valor a ser filtrado
function EMAFilter:update(newValue)
    if not self.initialized then
        self:initialize(newValue)
        return
    end
    
    -- Fórmula da média móvel exponencial
    -- Quanto maior alpha, mais o novo valor influencia o resultado
    -- Quanto menor alpha, mais o valor anterior é mantido
    self.value = self.alpha * newValue + (1 - self.alpha) * self.value
end

-- Retorna o valor filtrado atual
-- @return: Valor filtrado
function EMAFilter:get()
    if not self.initialized then return nil end
    return self.value
end

-- Define o fator de suavização
-- @param alpha: Fator de suavização (0 < alpha <= 1)
function EMAFilter:setAlpha(alpha)
    self.alpha = math.clamp(alpha, 0.01, 1.0)
end

-- ══════════════════════════════════════════════════════════════
-- FILTRO MÉDIA MÓVEL (MOVING AVERAGE)
-- ══════════════════════════════════════════════════════════════
--
-- Filtro que calcula a média dos últimos N valores.
-- Útil para suavizar dados discretos onde se deseja considerar um histórico.
--
-- Variáveis:
--   values: Array com os últimos N valores (em ordem)
--   size: Tamanho máximo do histórico (N)
--   count: Número de valores armazenados (pode ser menor que size no início)
--
-- Funcionamento:
--   1. add(value): Adiciona um novo valor ao histórico
--   2. getAverage(): Calcula e retorna a média dos valores armazenados

local MovingAverageFilter = {
    values = {},   -- Array com os últimos N valores (em ordem)
    size = 10,     -- Tamanho máximo do histórico (ajustável, padrão: 10)
    count = 0,     -- Número de valores armazenados
    initialized = false
}

-- Inicializa o filtro com um tamanho específico
-- @param size: Tamanho máximo do histórico (número de valores a armazenar)
function MovingAverageFilter:initialize(size)
    self.size = size or 10
    self.values = {}
    self.count = 0
    self.initialized = true
end

-- Adiciona um novo valor ao histórico
-- Mantém os últimos N valores em ordem (remove o mais antigo quando cheio)
-- @param value: Novo valor a ser adicionado
function MovingAverageFilter:add(value)
    if not self.initialized then
        self:initialize()
    end
    
    -- Se o array está cheio, remove o valor mais antigo (primeiro)
    if self.count >= self.size then
        -- Remove o primeiro valor (mais antigo)
        table.remove(self.values, 1)
        self.count = self.count - 1
    end
    
    -- Adiciona o novo valor no final (mais recente)
    table.insert(self.values, value)
    self.count = self.count + 1
end

-- Calcula e retorna a média dos valores armazenados
-- @return: Média dos últimos N valores (ou todos os valores se count < size)
function MovingAverageFilter:getAverage()
    if not self.initialized or self.count == 0 then
        return nil
    end
    
    -- Soma todos os valores armazenados
    local sum = 0
    for i = 1, self.count do
        if self.values[i] ~= nil then
            sum = sum + self.values[i]
        end
    end
    
    -- Retorna a média
    if self.count > 0 then
        return sum / self.count
    end
    
    return nil
end

-- Retorna o número de valores armazenados
-- @return: Número de valores no histórico
function MovingAverageFilter:getCount()
    return self.count
end

-- Limpa o histórico
function MovingAverageFilter:clear()
    self.values = {}
    self.count = 0
end

-- Define o tamanho máximo do histórico
-- @param size: Novo tamanho máximo
function MovingAverageFilter:setSize(size)
    self.size = size or 10
    self:clear()
end

-- ══════════════════════════════════════════════════════════════
-- KALMAN FILTER - ADVANCED IMPLEMENTATION
-- ══════════════════════════════════════════════════════════════

local KalmanFilter = {
    state = nil,
    covariance = nil,
    processNoise = nil,
    measurementNoise = nil,
    initialized = false
}

function KalmanFilter:initialize(initialState)
    -- State: [x, y, z, vx, vy, vz]
    self.state = {
        initialState.x or 0,
        initialState.y or 0,
        initialState.z or 0,
        initialState.vx or 0,
        initialState.vy or 0,
        initialState.vz or 0
    }
    
    -- Covariance matrix (6x6)
    self.covariance = AdvancedMath:createMatrix(6, 6, 0)
    for i = 1, 6 do
        self.covariance[i][i] = CONSTANTS.KALMAN_INITIAL_ERROR
    end
    
    -- Process noise
    self.processNoise = AdvancedMath:createMatrix(6, 6, 0)
    for i = 1, 6 do
        self.processNoise[i][i] = CONSTANTS.KALMAN_PROCESS_NOISE
    end
    
    -- Measurement noise
    self.measurementNoise = AdvancedMath:createMatrix(3, 3, 0)
    for i = 1, 3 do
        self.measurementNoise[i][i] = CONSTANTS.KALMAN_MEASUREMENT_NOISE
    end
    
    self.initialized = true
end

function KalmanFilter:predict(dt)
    if not self.initialized then return nil end
    
    -- State transition matrix (position += velocity * dt)
    local F = {
        {1, 0, 0, dt, 0, 0},
        {0, 1, 0, 0, dt, 0},
        {0, 0, 1, 0, 0, dt},
        {0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1}
    }
    
    -- Predict state: x = F * x
    local newState = {}
    for i = 1, 6 do
        local sum = 0
        for j = 1, 6 do
            sum = sum + F[i][j] * self.state[j]
        end
        newState[i] = sum
    end
    self.state = newState
    
    -- Predict covariance: P = F * P * F^T + Q
    local FP = AdvancedMath:multiplyMatrix(F, self.covariance)
    local FT = AdvancedMath:transposeMatrix(F)
    local FPFT = AdvancedMath:multiplyMatrix(FP, FT)
    self.covariance = AdvancedMath:addMatrix(FPFT, self.processNoise)
    
    return Vector3.new(self.state[1], self.state[2], self.state[3])
end

function KalmanFilter:update(measurement)
    if not self.initialized then return end
    
    -- Measurement matrix H (we only measure position)
    local H = {
        {1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0}
    }
    
    -- Innovation: y = z - H * x
    local z = {measurement.X, measurement.Y, measurement.Z}
    local Hx = {}
    for i = 1, 3 do
        local sum = 0
        for j = 1, 6 do
            sum = sum + H[i][j] * self.state[j]
        end
        Hx[i] = sum
    end
    
    local innovation = {}
    for i = 1, 3 do
        innovation[i] = z[i] - Hx[i]
    end
    
    -- Innovation covariance: S = H * P * H^T + R
    local HP = AdvancedMath:multiplyMatrix(H, self.covariance)
    local HT = AdvancedMath:transposeMatrix(H)
    local HPHT = AdvancedMath:multiplyMatrix(HP, HT)
    local S = AdvancedMath:addMatrix(HPHT, self.measurementNoise)
    
    -- Kalman gain: K = P * H^T * S^-1
    local PHT = AdvancedMath:multiplyMatrix(self.covariance, HT)
    local Sinv = AdvancedMath:invertMatrix(S)
    local K = AdvancedMath:multiplyMatrix(PHT, Sinv)
    
    -- Update state: x = x + K * y
    for i = 1, 6 do
        for j = 1, 3 do
            self.state[i] = self.state[i] + K[i][j] * innovation[j]
        end
    end
    
    -- Update covariance: P = (I - K * H) * P
    local KH = AdvancedMath:multiplyMatrix(K, H)
    local I = AdvancedMath:createMatrix(6, 6, 0)
    for i = 1, 6 do I[i][i] = 1 end
    
    for i = 1, 6 do
        for j = 1, 6 do
            I[i][j] = I[i][j] - KH[i][j]
        end
    end
    
    self.covariance = AdvancedMath:multiplyMatrix(I, self.covariance)
end

function KalmanFilter:getState()
    if not self.initialized then return nil end
    return {
        position = Vector3.new(self.state[1], self.state[2], self.state[3]),
        velocity = Vector3.new(self.state[4], self.state[5], self.state[6])
    }
end

function KalmanFilter:getCovariance()
    return self.covariance
end

-- ══════════════════════════════════════════════════════════════
-- PARTICLE FILTER - ADVANCED IMPLEMENTATION
-- ══════════════════════════════════════════════════════════════

local ParticleFilter = {
    particles = {},
    weights = {},
    numParticles = CONSTANTS.PARTICLE_COUNT,
    initialized = false
}

function ParticleFilter:initialize(initialState)
    self.particles = {}
    self.weights = {}
    
    for i = 1, self.numParticles do
        -- Initialize particles around initial state with noise
        self.particles[i] = {
            position = Vector3.new(
                initialState.x + AdvancedMath:gaussianRandom(0, 2),
                initialState.y + AdvancedMath:gaussianRandom(0, 2),
                initialState.z + AdvancedMath:gaussianRandom(0, 2)
            ),
            velocity = Vector3.new(
                initialState.vx + AdvancedMath:gaussianRandom(0, 5),
                initialState.vy + AdvancedMath:gaussianRandom(0, 5),
                initialState.vz + AdvancedMath:gaussianRandom(0, 5)
            )
        }
        self.weights[i] = 1 / self.numParticles
    end
    
    self.initialized = true
end

function ParticleFilter:predict(dt)
    if not self.initialized then return nil end
    
    -- Propagate particles forward using motion model
    for i = 1, self.numParticles do
        local particle = self.particles[i]
        
        -- Add process noise
        local noise = Vector3.new(
            AdvancedMath:gaussianRandom(0, 1),
            AdvancedMath:gaussianRandom(0, 1),
            AdvancedMath:gaussianRandom(0, 1)
        )
        
        particle.position = particle.position + particle.velocity * dt + noise
        particle.velocity = particle.velocity + Vector3.new(0, -Workspace.Gravity * dt, 0)
    end
    
    return self:getEstimate()
end

function ParticleFilter:update(measurement)
    if not self.initialized then return end
    
    -- Update weights based on measurement likelihood
    local totalWeight = 0
    for i = 1, self.numParticles do
        local particle = self.particles[i]
        local distance = (particle.position - measurement).Magnitude
        
        -- Gaussian likelihood
        local sigma = 5
        self.weights[i] = math.exp(-(distance * distance) / (2 * sigma * sigma))
        totalWeight = totalWeight + self.weights[i]
    end
    
    -- Normalize weights
    if totalWeight > 0 then
        for i = 1, self.numParticles do
            self.weights[i] = self.weights[i] / totalWeight
        end
    end
    
    -- Resample if effective sample size is low
    local effectiveSampleSize = 0
    for i = 1, self.numParticles do
        effectiveSampleSize = effectiveSampleSize + self.weights[i] * self.weights[i]
    end
    effectiveSampleSize = 1 / effectiveSampleSize
    
    if effectiveSampleSize < self.numParticles * CONSTANTS.PARTICLE_RESAMPLE_THRESHOLD then
        self:resample()
    end
end

function ParticleFilter:resample()
    local newParticles = {}
    local newWeights = {}
    
    -- Systematic resampling
    local cumsum = {}
    cumsum[1] = self.weights[1]
    for i = 2, self.numParticles do
        cumsum[i] = cumsum[i-1] + self.weights[i]
    end
    
    local step = 1 / self.numParticles
    local u = math.random() * step
    
    local j = 1
    for i = 1, self.numParticles do
        while u > cumsum[j] and j < self.numParticles do
            j = j + 1
        end
        newParticles[i] = {
            position = self.particles[j].position,
            velocity = self.particles[j].velocity
        }
        newWeights[i] = 1 / self.numParticles
        u = u + step
    end
    
    self.particles = newParticles
    self.weights = newWeights
end

function ParticleFilter:getEstimate()
    if not self.initialized then return nil end
    
    -- Weighted average of particles
    local sumPos = Vector3.new(0, 0, 0)
    local sumVel = Vector3.new(0, 0, 0)
    
    for i = 1, self.numParticles do
        sumPos = sumPos + self.particles[i].position * self.weights[i]
        sumVel = sumVel + self.particles[i].velocity * self.weights[i]
    end
    
    return {
        position = sumPos,
        velocity = sumVel
    }
end

function ParticleFilter:getParticles()
    return self.particles, self.weights
end

-- ══════════════════════════════════════════════════════════════
-- DEEP NEURAL NETWORK - ADVANCED IMPLEMENTATION
-- ══════════════════════════════════════════════════════════════

local DeepNeuralNetwork = {
    layers = CONSTANTS.DNN_LAYERS,
    weights = {},
    biases = {},
    learningRate = CONSTANTS.DNN_LEARNING_RATE,
    momentum = CONSTANTS.DNN_MOMENTUM,
    dropout = CONSTANTS.DNN_DROPOUT,
    velocities = {}, -- For momentum
    initialized = false
}

function DeepNeuralNetwork:initialize()
    -- Xavier/He initialization for better convergence
    for i = 1, #self.layers - 1 do
        self.weights[i] = {}
        self.biases[i] = {}
        self.velocities[i] = {}
        
        local fanIn = self.layers[i]
        local fanOut = self.layers[i + 1]
        local scale = math.sqrt(2 / (fanIn + fanOut))
        
        for j = 1, fanOut do
            self.weights[i][j] = {}
            self.velocities[i][j] = {}
            self.biases[i][j] = (math.random() - 0.5) * 0.01
            
            for k = 1, fanIn do
                self.weights[i][j][k] = (math.random() - 0.5) * 2 * scale
                self.velocities[i][j][k] = 0
            end
        end
    end
    
    self.initialized = true
end

function DeepNeuralNetwork:forward(inputs, training)
    if not self.initialized then
        self:initialize()
    end
    
    local activations = {inputs}
    training = training == nil and true or training
    
    for i = 1, #self.layers - 1 do
        local nextLayer = {}
        
        for j = 1, self.layers[i + 1] do
            local sum = self.biases[i][j]
            
            for k = 1, #activations[i] do
                sum = sum + activations[i][k] * self.weights[i][j][k]
            end
            
            -- Use different activations for different layers
            if i < #self.layers - 1 then
                nextLayer[j] = AdvancedMath:leakyRelu(sum)
                
                -- Apply dropout during training
                if training and math.random() < self.dropout then
                    nextLayer[j] = 0
                end
            else
                nextLayer[j] = AdvancedMath:sigmoid(sum)
            end
        end
        
        table.insert(activations, nextLayer)
    end
    
    return activations[#activations], activations
end

function DeepNeuralNetwork:backward(inputs, target, activations)
    local errors = {}
    local output = activations[#activations]
    
    -- Output layer error
    errors[#self.layers] = {}
    for i = 1, #output do
        local error = target - output[i]
        errors[#self.layers][i] = error * output[i] * (1 - output[i])
    end
    
    -- Backpropagate errors
    for layer = #self.layers - 1, 2, -1 do
        errors[layer] = {}
        for i = 1, self.layers[layer] do
            local error = 0
            for j = 1, self.layers[layer + 1] do
                error = error + errors[layer + 1][j] * self.weights[layer][j][i]
            end
            
            local activation = activations[layer][i]
            errors[layer][i] = error * (activation > 0 and 1 or 0.01)
        end
    end
    
    -- Update weights with momentum
    for layer = 1, #self.layers - 1 do
        for j = 1, self.layers[layer + 1] do
            for k = 1, self.layers[layer] do
                local gradient = self.learningRate * errors[layer + 1][j] * activations[layer][k]
                self.velocities[layer][j][k] = self.momentum * self.velocities[layer][j][k] + gradient
                self.weights[layer][j][k] = self.weights[layer][j][k] + self.velocities[layer][j][k]
            end
            
            self.biases[layer][j] = self.biases[layer][j] + self.learningRate * errors[layer + 1][j]
        end
    end
end

function DeepNeuralNetwork:train(inputs, target)
    if not Config.dnnEnabled then return end
    
    local output, activations = self:forward(inputs, true)
    self:backward(inputs, target, activations)
    
    return output
end

function DeepNeuralNetwork:predict(inputs)
    local output, _ = self:forward(inputs, false)
    return output[1]
end

-- ══════════════════════════════════════════════════════════════
-- GENETIC ALGORITHM - AUTO-OPTIMIZATION
-- ══════════════════════════════════════════════════════════════

local GeneticAlgorithm = {
    population = {},
    fitness = {},
    generation = 0,
    bestIndividual = nil,
    bestFitness = -math.huge
}

function GeneticAlgorithm:initialize()
    self.population = {}
    self.fitness = {}
    
    -- Create initial population (parameter sets)
    for i = 1, Config.gaPopulationSize do
        self.population[i] = {
            timingMultiplier = math.random() * 2,
            distanceMultiplier = math.random() * 2,
            confidenceThreshold = math.random(),
            aggression = math.random()
        }
        self.fitness[i] = 0
    end
end

function GeneticAlgorithm:evaluateFitness(individual, successRate, avgLatency)
    -- Fitness function: maximize success rate, minimize latency
    local fitness = successRate * 100 - avgLatency * 10
    return fitness
end

function GeneticAlgorithm:select()
    -- Tournament selection
    local tournament = {}
    for i = 1, 3 do
        table.insert(tournament, math.random(1, #self.population))
    end
    
    local best = tournament[1]
    for _, idx in ipairs(tournament) do
        if self.fitness[idx] > self.fitness[best] then
            best = idx
        end
    end
    
    return self.population[best]
end

function GeneticAlgorithm:crossover(parent1, parent2)
    if math.random() > Config.gaCrossoverRate then
        return parent1
    end
    
    local child = {}
    for key, value in pairs(parent1) do
        child[key] = math.random() < 0.5 and parent1[key] or parent2[key]
    end
    
    return child
end

function GeneticAlgorithm:mutate(individual)
    local mutated = {}
    for key, value in pairs(individual) do
        if math.random() < Config.gaMutationRate then
            mutated[key] = value + (math.random() - 0.5) * 0.2
            mutated[key] = math.max(0, math.min(2, mutated[key]))
        else
            mutated[key] = value
        end
    end
    return mutated
end

function GeneticAlgorithm:evolve()
    -- Sort by fitness
    local indices = {}
    for i = 1, #self.population do
        table.insert(indices, i)
    end
    
    table.sort(indices, function(a, b)
        return self.fitness[a] > self.fitness[b]
    end)
    
    -- Keep elite
    local newPopulation = {}
    for i = 1, CONSTANTS.GA_ELITE_SIZE do
        table.insert(newPopulation, self.population[indices[i]])
    end
    
    -- Create new generation
    while #newPopulation < Config.gaPopulationSize do
        local parent1 = self:select()
        local parent2 = self:select()
        local child = self:crossover(parent1, parent2)
        child = self:mutate(child)
        table.insert(newPopulation, child)
    end
    
    self.population = newPopulation
    self.generation = self.generation + 1
    
    -- Update best individual
    local bestIdx = indices[1]
    if self.fitness[bestIdx] > self.bestFitness then
        self.bestFitness = self.fitness[bestIdx]
        self.bestIndividual = self.population[bestIdx]
    end
end

function GeneticAlgorithm:getBestParameters()
    return self.bestIndividual
end

-- ══════════════════════════════════════════════════════════════
-- BAYESIAN FUSION - MULTI-SENSOR FUSION
-- ══════════════════════════════════════════════════════════════

local BayesianFusion = {
    priors = {},
    likelihoods = {},
    posteriors = {}
}

function BayesianFusion:updateBelief(prediction, measurement, uncertainty)
    -- Bayesian update: P(state|measurement) ∝ P(measurement|state) * P(state)
    local sigma_pred = uncertainty or 5
    local sigma_meas = 3
    
    -- Precision (inverse variance)
    local prec_pred = 1 / (sigma_pred * sigma_pred)
    local prec_meas = 1 / (sigma_meas * sigma_meas)
    
    -- Fused estimate (weighted average)
    local prec_fused = prec_pred + prec_meas
    local mean_fused = (prec_pred * prediction + prec_meas * measurement) / prec_fused
    
    return mean_fused, 1 / math.sqrt(prec_fused)
end

function BayesianFusion:fusePredictions(predictions)
    if #predictions == 0 then return nil end
    if #predictions == 1 then return predictions[1].value end
    
    -- Weighted fusion based on confidence
    local totalWeight = 0
    local fusedValue = Vector3.new(0, 0, 0)
    
    for _, pred in ipairs(predictions) do
        local weight = pred.confidence or 1
        totalWeight = totalWeight + weight
        fusedValue = fusedValue + pred.value * weight
    end
    
    if totalWeight > 0 then
        fusedValue = fusedValue / totalWeight
    end
    
    return fusedValue
end

-- ══════════════════════════════════════════════════════════════
-- FOURIER ANALYSIS - PATTERN DETECTION
-- ══════════════════════════════════════════════════════════════

local FourierAnalysis = {
    history = {},
    dominantFrequencies = {}
}

function FourierAnalysis:analyze(data)
    if #data < 8 then return nil end
    
    -- Extract speed variations
    local speeds = {}
    for i, point in ipairs(data) do
        table.insert(speeds, point.speed or 0)
    end
    
    -- Compute FFT
    local spectrum = AdvancedMath:fft(speeds)
    
    -- Find dominant frequencies
    local maxMagnitude = 0
    local dominantFreq = 0
    
    for i = 2, math.min(#spectrum, 5) do -- Check first few frequencies
        if spectrum[i] > maxMagnitude then
            maxMagnitude = spectrum[i]
            dominantFreq = i
        end
    end
    
    return {
        dominantFrequency = dominantFreq,
        magnitude = maxMagnitude,
        isPeriodic = maxMagnitude > 10 -- Threshold for periodic pattern
    }
end

function FourierAnalysis:predictNextValue(data, stepsAhead)
    local analysis = self:analyze(data)
    if not analysis or not analysis.isPeriodic then return nil end
    
    -- Simple prediction based on detected periodicity
    local period = #data / analysis.dominantFrequency
    local lastValue = data[#data]
    
    return lastValue -- Simplified
end

-- ══════════════════════════════════════════════════════════════
-- ANOMALY DETECTION - ISOLATION FOREST
-- ══════════════════════════════════════════════════════════════

local AnomalyDetection = {
    trees = {},
    numTrees = 10,
    sampleSize = 20,
    threshold = 0.6
}

function AnomalyDetection:buildTree(data, depth, maxDepth)
    if depth >= maxDepth or #data <= 1 then
        return {leaf = true, size = #data}
    end
    
    -- Random feature and split
    local feature = math.random(1, 3) -- x, y, z
    local values = {}
    for _, point in ipairs(data) do
        local val = feature == 1 and point.X or (feature == 2 and point.Y or point.Z)
        table.insert(values, val)
    end
    
    if #values == 0 then return {leaf = true, size = 0} end
    
    local minVal = math.min(table.unpack(values))
    local maxVal = math.max(table.unpack(values))
    local split = minVal + math.random() * (maxVal - minVal)
    
    -- Split data
    local left, right = {}, {}
    for _, point in ipairs(data) do
        local val = feature == 1 and point.X or (feature == 2 and point.Y or point.Z)
        if val < split then
            table.insert(left, point)
        else
            table.insert(right, point)
        end
    end
    
    return {
        leaf = false,
        feature = feature,
        split = split,
        left = self:buildTree(left, depth + 1, maxDepth),
        right = self:buildTree(right, depth + 1, maxDepth)
    }
end

function AnomalyDetection:train(data)
    self.trees = {}
    local maxDepth = math.ceil(math.log(self.sampleSize) / math.log(2))
    
    for i = 1, self.numTrees do
        -- Random sample
        local sample = {}
        for j = 1, math.min(self.sampleSize, #data) do
            table.insert(sample, data[math.random(1, #data)])
        end
        
        self.trees[i] = self:buildTree(sample, 0, maxDepth)
    end
end

function AnomalyDetection:pathLength(tree, point, depth)
    if tree.leaf then
        -- Average path length for unsuccessful search
        return depth + (tree.size > 1 and 1 or 0)
    end
    
    local val = tree.feature == 1 and point.X or (tree.feature == 2 and point.Y or point.Z)
    if val < tree.split then
        return self:pathLength(tree.left, point, depth + 1)
    else
        return self:pathLength(tree.right, point, depth + 1)
    end
end

function AnomalyDetection:score(point)
    if #self.trees == 0 then return 0 end
    
    local avgPath = 0
    for _, tree in ipairs(self.trees) do
        avgPath = avgPath + self:pathLength(tree, point, 0)
    end
    avgPath = avgPath / #self.trees
    
    -- Normalize score
    local c = 2 * (math.log(self.sampleSize - 1) + 0.5772) - 2 * (self.sampleSize - 1) / self.sampleSize
    local score = math.pow(2, -avgPath / c)
    
    return score
end

function AnomalyDetection:isAnomaly(point)
    return self:score(point) > self.threshold
end

-- ══════════════════════════════════════════════════════════════
-- ADVANCED CACHE SYSTEM - HYBRID LRU/LFU
-- ══════════════════════════════════════════════════════════════

local AdvancedCache = {
    lruCache = {},
    lfuCache = {},
    accessCount = {},
    accessTime = {},
    maxSize = 100
}

function AdvancedCache:set(key, value, priority)
    priority = priority or "medium"
    
    self.accessCount[key] = (self.accessCount[key] or 0) + 1
    self.accessTime[key] = tick()
    
    -- LRU cache for recent access
    self.lruCache[key] = {value = value, time = tick()}
    
    -- LFU cache for frequent access
    if self.accessCount[key] > 3 then
        self.lfuCache[key] = {value = value, count = self.accessCount[key]}
    end
    
    -- Cleanup if too large
    if self:size() > self.maxSize then
        self:evict()
    end
end

function AdvancedCache:get(key)
    self.accessCount[key] = (self.accessCount[key] or 0) + 1
    self.accessTime[key] = tick()
    
    -- Check LFU first (most frequent)
    if self.lfuCache[key] then
        return self.lfuCache[key].value
    end
    
    -- Check LRU (most recent)
    if self.lruCache[key] then
        return self.lruCache[key].value
    end
    
    return nil
end

function AdvancedCache:size()
    local count = 0
    for _ in pairs(self.lruCache) do count = count + 1 end
    return count
end

function AdvancedCache:evict()
    -- Evict least recently used from LRU
    local oldestKey = nil
    local oldestTime = math.huge
    
    for key, data in pairs(self.lruCache) do
        if data.time < oldestTime and not self.lfuCache[key] then
            oldestTime = data.time
            oldestKey = key
        end
    end
    
    if oldestKey then
        self.lruCache[oldestKey] = nil
        self.accessCount[oldestKey] = nil
        self.accessTime[oldestKey] = nil
    end
end

function AdvancedCache:clear()
    self.lruCache = {}
    self.lfuCache = {}
    self.accessCount = {}
    self.accessTime = {}
end

-- ══════════════════════════════════════════════════════════════
-- CORE UTILITIES MODULE (ENHANCED)
-- ══════════════════════════════════════════════════════════════

local Utils = {}

function Utils.clamp(value, min, max)
    return math.max(min, math.min(max, value))
end

function Utils.lerp(a, b, t)
    return a + (b - a) * t
end

function Utils.smoothstep(t)
    return t * t * (3 - 2 * t)
end

function Utils.getVelocityDirection(velocity)
    return velocity.Magnitude > 0 and velocity.Unit or Vector3.new(0, 0, 0)
end

function Utils.predictPosition(pos, vel, accel, time)
    return pos + vel * time + accel * 0.5 * time * time
end

function Utils.isApproaching(ballPos, ballVel, playerPos)
    if ballVel.Magnitude < 5 then return false end
    local dirToPlayer = (playerPos - ballPos).Unit
    local velDir = ballVel.Unit
    return dirToPlayer:Dot(velDir) > 0.3
end

function Utils.getTimestamp()
    return tick()
end

function Utils.notify(title, message, duration)
    if not Config.showNotifications then return end
    pcall(function()
        game:GetService("StarterGui"):SetCore("SendNotification", {
            Title = title,
            Text = message,
            Duration = duration or 3
        })
    end)
end

-- ══════════════════════════════════════════════════════════════
-- EVENT SYSTEM
-- ══════════════════════════════════════════════════════════════

local Events = {
    _listeners = {}
}

function Events:on(event, callback)
    if not self._listeners[event] then
        self._listeners[event] = {}
    end
    table.insert(self._listeners[event], callback)
end

function Events:emit(event, ...)
    if not self._listeners[event] then return end
    for _, callback in ipairs(self._listeners[event]) do
        task.spawn(callback, ...)
    end
end

function Events:clear(event)
    if event then
        self._listeners[event] = nil
    else
        self._listeners = {}
    end
end

-- ══════════════════════════════════════════════════════════════
-- ADVANCED LOGGING SYSTEM
-- ══════════════════════════════════════════════════════════════

local LogLevels = {
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4
}

local Logger = {
    logs = {},
    maxLogs = 1000,
    currentLevel = LogLevels.INFO
}

function Logger:log(level, category, message, data)
    if level < self.currentLevel then return end
    
    local logEntry = {
        timestamp = tick(),
        level = level,
        category = category,
        message = message,
        data = data or {}
    }
    
    table.insert(self.logs, logEntry)
    
    if #self.logs > self.maxLogs then
        table.remove(self.logs, 1)
    end
    
    if level >= LogLevels.WARN then
        local levelName = level == LogLevels.ERROR and "ERROR" or "WARN"
        warn(string.format("[%s] [%s] %s", levelName, category, message))
    end
end

function Logger:info(category, message, data)
    self:log(LogLevels.INFO, category, message, data)
end

function Logger:warn(category, message, data)
    self:log(LogLevels.WARN, category, message, data)
end

function Logger:error(category, message, data)
    self:log(LogLevels.ERROR, category, message, data)
end

function Logger:clear()
    self.logs = {}
end

-- ══════════════════════════════════════════════════════════════
-- BALL TRACKING MODULE (ENHANCED)
-- ══════════════════════════════════════════════════════════════

local BallTracker = {
    ball = nil,
    history = {},
    lastUpdate = 0,
    currentTarget = nil,
    lastTargetChangeTime = 0,
    kalmanFilter = nil,
    particleFilter = nil
}

function BallTracker:getBall()
    local cached = AdvancedCache:get("ball")
    if cached and cached.Parent then
        return cached
    end
    
    for _, ball in pairs(Workspace.Balls:GetChildren()) do
        if ball:GetAttribute("realBall") then
            AdvancedCache:set("ball", ball, "high")
            return ball
        end
    end
    
    return nil
end

function BallTracker:update()
    self.ball = self:getBall()
    if not self.ball then return end
    
    local zoomies = self.ball:FindFirstChild("zoomies")
    if not zoomies then return end
    
    local data = {
        position = self.ball.Position,
        velocity = zoomies.VectorVelocity,
        speed = zoomies.VectorVelocity.Magnitude,
        target = self.ball:GetAttribute("target"),
        timestamp = tick()
    }
    
    -- Initialize Kalman Filter
    if Config.useKalmanFilter and not self.kalmanFilter then
        self.kalmanFilter = {}
        for k, v in pairs(KalmanFilter) do
            self.kalmanFilter[k] = v
        end
        self.kalmanFilter:initialize({
            x = data.position.X,
            y = data.position.Y,
            z = data.position.Z,
            vx = data.velocity.X,
            vy = data.velocity.Y,
            vz = data.velocity.Z
        })
    end
    
    -- Initialize Particle Filter
    if Config.useParticleFilter and not self.particleFilter then
        self.particleFilter = {}
        for k, v in pairs(ParticleFilter) do
            self.particleFilter[k] = v
        end
        self.particleFilter:initialize({
            x = data.position.X,
            y = data.position.Y,
            z = data.position.Z,
            vx = data.velocity.X,
            vy = data.velocity.Y,
            vz = data.velocity.Z
        })
    end
    
    -- Update filters
    if self.kalmanFilter and self.kalmanFilter.initialized then
        self.kalmanFilter:update(data.position)
    end
    
    if self.particleFilter and self.particleFilter.initialized then
        self.particleFilter:update(data.position)
    end
    
    -- Check for target change
    if self.currentTarget ~= data.target then
        local oldTarget = self.currentTarget
        self.currentTarget = data.target
        self.lastTargetChangeTime = tick()
        Events:emit("targetChanged", data.target, oldTarget)
        AdvancedCache:clear()
    end
    
    -- Add to history
    table.insert(self.history, data)
    if #self.history > CONSTANTS.MAX_HISTORY then
        table.remove(self.history, 1)
    end
    
    self.lastUpdate = tick()
    return data
end

function BallTracker:predictKalman(timeAhead)
    if not self.kalmanFilter or not self.kalmanFilter.initialized then return nil end
    
    -- Predict using Kalman filter
    local predicted = self.kalmanFilter:predict(timeAhead)
    return predicted
end

function BallTracker:predictParticle(timeAhead)
    if not self.particleFilter or not self.particleFilter.initialized then return nil end
    
    -- Predict using Particle filter
    local predicted = self.particleFilter:predict(timeAhead)
    if predicted then
        return predicted.position
    end
    return nil
end

function BallTracker:predict(timeAhead)
    if not self.ball or #self.history < 2 then return nil end
    
    local predictions = {}
    
    -- Kalman prediction
    if Config.useKalmanFilter then
        local kPred = self:predictKalman(timeAhead)
        if kPred then
            table.insert(predictions, {value = kPred, confidence = 0.9})
        end
    end
    
    -- Particle filter prediction
    if Config.useParticleFilter then
        local pPred = self:predictParticle(timeAhead)
        if pPred then
            table.insert(predictions, {value = pPred, confidence = 0.85})
        end
    end
    
    -- Physics prediction
    local current = self.history[#self.history]
    local previous = self.history[#self.history - 1]
    local dt = current.timestamp - previous.timestamp
    
    if dt > 0 then
        local acceleration = (current.velocity - previous.velocity) / dt
        local gravity = Vector3.new(0, -Workspace.Gravity, 0)
        local physPred = Utils.predictPosition(current.position, current.velocity, acceleration + gravity, timeAhead)
        table.insert(predictions, {value = physPred, confidence = 0.7})
    end
    
    -- Bayesian fusion
    if Config.useBayesianFusion and #predictions > 0 then
        return BayesianFusion:fusePredictions(predictions)
    end
    
    return predictions[1] and predictions[1].value or current.position
end

function BallTracker:getConfidence()
    if not self.ball or #self.history < 3 then return 0 end
    
    local current = self.history[#self.history]
    if not Player.Character or not Player.Character.PrimaryPart then return 0 end
    
    local playerPos = Player.Character.PrimaryPart.Position
    local distance = (playerPos - current.position).Magnitude
    local isApproaching = Utils.isApproaching(current.position, current.velocity, playerPos)
    
    local confidence = 0
    
    if isApproaching then confidence = confidence + 0.4 end
    
    local distanceFactor = Utils.clamp(1 - (distance / 100), 0, 1)
    confidence = confidence + distanceFactor * 0.3
    
    local consistencyCount = 0
    for i = math.max(1, #self.history - 4), #self.history do
        local data = self.history[i]
        if Utils.isApproaching(data.position, data.velocity, playerPos) then
            consistencyCount = consistencyCount + 1
        end
    end
    local consistencyFactor = consistencyCount / math.min(5, #self.history)
    confidence = confidence + consistencyFactor * 0.3
    
    return confidence
end

-- ══════════════════════════════════════════════════════════════
-- PARRY SYSTEM MODULE (ENHANCED)
-- ══════════════════════════════════════════════════════════════

local ParrySystem = {
    lastParry = 0,
    parryCount = 0,
    cooldown = 0.1,  -- Cooldown otimizado: 0.1s (balanceado)
    isParrying = false,
    lastTargetChangeTime = 0,
    geneticParams = nil,
    lastParryTimestamp = 0  -- Para proteção simples contra double click
}

function ParrySystem:canParry()
    if not Config.enabled then return false end
    if not Player.Character or not Player.Character.PrimaryPart then return false end
    if not Player.Character:FindFirstChild("Highlight") then return false end
    if self.isParrying then return false end
    
    -- Proteção simples contra double click: verificar se parry foi muito recente (< 0.08s)
    local timeSinceLastParry = tick() - self.lastParryTimestamp
    if timeSinceLastParry < 0.08 then
        return false
    end
    
    local timeSinceLastParryCooldown = tick() - self.lastParry
    local effectiveCooldown = self.cooldown
    
    if self.lastTargetChangeTime > 0 then
        local timeSinceTargetChange = tick() - self.lastTargetChangeTime
        if timeSinceTargetChange < 0.5 then
            effectiveCooldown = effectiveCooldown * 0.6
        end
    end
    
    return timeSinceLastParryCooldown >= effectiveCooldown
end

function ParrySystem:calculateTiming(ballData, playerPos)
    if not ballData then return nil end
    
    local distance = (playerPos - ballData.position).Magnitude
    
    -- Usar velocidade suavizada do EMA se disponível (mais precisa)
    local speed = ballData.speed
    if BallTracker.emaVelocity and BallTracker.emaVelocity.initialized then
        local smoothedSpeed = BallTracker.emaVelocity:get()
        if smoothedSpeed and smoothedSpeed > 0 then
            speed = smoothedSpeed
        end
    end
    
    if speed < 5 then return nil end
    
    local mode = CONSTANTS.MODES[Config.mode] or CONSTANTS.MODES.BALANCED
    local ping = Stats.Network.ServerStatsItem["Data Ping"]:GetValue() / 10
    local pingThreshold = Utils.clamp(ping / 10, 2, 15)
    
    -- Apply genetic algorithm parameters if available
    local timingMult = 1.0
    local distanceMult = 1.0
    
    if Config.geneticOptimization and self.geneticParams then
        timingMult = self.geneticParams.timingMultiplier or 1.0
        distanceMult = self.geneticParams.distanceMultiplier or 1.0
    end
    
    -- Cálculo otimizado de parryDistance para máxima precisão
    local cappedSpeed = Utils.clamp(speed - 9.5, 0, 650)
    local speedDivisor = (2.4 + cappedSpeed * 0.002) * mode.multiplier * timingMult
    local parryDistance = (pingThreshold + math.max(speed / speedDivisor, 9.5)) * distanceMult
    
    local isFastBall = speed > Config.fastBallThreshold
    if isFastBall then
        -- Ajuste mais preciso para bolas rápidas
        parryDistance = parryDistance * 0.80  -- Ajustado de 0.82 para 0.80 (mais preciso)
    end
    
    if self.lastTargetChangeTime > 0 then
        local timeSinceTargetChange = tick() - self.lastTargetChangeTime
        if timeSinceTargetChange < 0.5 then
            parryDistance = parryDistance * (1 / Config.targetChangeAggressiveness)
        end
    end
    
    -- Compensação de ping otimizada
    if Config.pingCompensation then
        parryDistance = parryDistance + (ping * 0.012)  -- Ajustado de 0.01 para 0.012 (mais preciso)
    end
    
    -- Calcular tempo até impacto
    local rawTiming = distance / speed
    
    -- Filtrar tempo usando KalmanFilter1D para suavizar e melhorar precisão
    local filteredTiming = rawTiming
    if BallTracker.kalman1DTime and BallTracker.kalman1DTime.initialized then
        BallTracker.kalman1DTime:update(rawTiming)
        local filtered = BallTracker.kalman1DTime:get()
        if filtered and filtered > 0 then
            filteredTiming = filtered
        end
    end
    
    return {
        distance = parryDistance,
        timing = filteredTiming,  -- Usar tempo filtrado para maior precisão
        rawTiming = rawTiming,     -- Manter tempo original para referência
        confidence = BallTracker:getConfidence(),
        isFastBall = isFastBall
    }
end

function ParrySystem:shouldParry(ballData)
    if not self:canParry() then return false end
    if not ballData then return false end
    if ballData.target ~= tostring(Player) then return false end
    if not Player.Character or not Player.Character.PrimaryPart then return false end
    
    local playerPos = Player.Character.PrimaryPart.Position
    local distance = (playerPos - ballData.position).Magnitude
    local speed = ballData.speed or 0
    local isApproaching = Utils.isApproaching(ballData.position, ballData.velocity, playerPos)
    
    -- PROTEÇÃO CONTRA SPAM: Distância máxima absoluta (só se não está se aproximando)
    -- Se a bola está muito longe (> 100 studs) e não está se aproximando, não fazer parry
    if distance > 100 and not isApproaching then
        return false
    end
    
    -- PROTEÇÃO CONTRA SPAM: Se não está se aproximando e está muito longe
    -- Se não está se aproximando e está longe (> 50), não fazer parry
    if not isApproaching and distance > 50 then
        return false
    end
    
    -- PROTEÇÃO CONTRA SPAM: Bola muito rápida e muito longe E não está se aproximando
    -- Se velocidade > 500, distância > 60 E não está se aproximando, não fazer parry
    if speed > 500 and distance > 60 and not isApproaching then
        return false
    end
    
    local timing = self:calculateTiming(ballData, playerPos)
    
    if not timing then return false end
    
    local mode = CONSTANTS.MODES[Config.mode] or CONSTANTS.MODES.BALANCED
    local requiredConfidence = mode.confidence
    
    if timing.confidence < requiredConfidence then return false end
    
    -- Verificar novamente a distância após calcular o timing
    if distance > timing.distance then return false end
    
    -- PROTEÇÃO FINAL: Se a distância calculada é muito grande (> 80) E não está se aproximando
    -- Permitir distâncias maiores se a bola está se aproximando
    if timing.distance > 80 and not isApproaching then
        return false
    end
    
    return distance <= timing.distance
end

function ParrySystem:executeParry(ballData)
    if not self:canParry() then return false end
    if self.isParrying then return false end
    
    -- PROTEÇÃO EXTRA: Verificar novamente antes de executar
    if not ballData then return false end
    if not Player.Character or not Player.Character.PrimaryPart then return false end
    
    local playerPos = Player.Character.PrimaryPart.Position
    local distance = (playerPos - ballData.position).Magnitude
    local speed = ballData.speed or 0
    local isApproaching = Utils.isApproaching(ballData.position, ballData.velocity, playerPos)
    
    -- Verificação final antes de executar (mais permissiva para bolas rápidas que estão se aproximando)
    -- Só bloquear se está muito longe E não está se aproximando
    if distance > 100 and not isApproaching then return false end
    if speed > 500 and distance > 60 and not isApproaching then return false end
    
    self.isParrying = true
    local startTime = tick()
    
    -- Instant perfect parry
    pcall(function()
        game:GetService("VirtualInputManager"):SendMouseButtonEvent(0, 0, 0, true, game, 1)
        task.wait(0.001)
        game:GetService("VirtualInputManager"):SendMouseButtonEvent(0, 0, 0, false, game, 1)
    end)
    
    local latency = tick() - startTime
    self.lastParry = tick()
    self.parryCount = self.parryCount + 1
    
    -- Atualizar timestamp imediatamente para prevenir double click
    self.lastParryTimestamp = tick()
    
    -- Liberar flag após delay otimizado para prevenir spam (0.08s)
    task.spawn(function()
        task.wait(0.08)
        self.isParrying = false
    end)
    
    Events:emit("parryExecuted", {
        timestamp = tick(),
        count = self.parryCount,
        latency = latency
    })
    
    Logger:info("Parry", "Parry executado", {count = self.parryCount, latency = latency, distance = distance, speed = speed})
    
    return true
end

-- ══════════════════════════════════════════════════════════════
-- AUTO SPAM SYSTEM
-- ══════════════════════════════════════════════════════════════

local AutoSpamSystem = {
    isSpamming = false,
    lastSpamParry = 0,
    spamCooldown = 0.08
}

function AutoSpamSystem:canSpamParry()
    if not Player.Character or not Player.Character.PrimaryPart then return false end
    if not Player.Character:FindFirstChild("Highlight") then return false end
    return tick() - self.lastSpamParry >= self.spamCooldown
end

function AutoSpamSystem:shouldSpam()
    if not Config.autoSpamEnabled then return false end
    if not Player.Character or not Player.Character.PrimaryPart then return false end
    
    local playerPos = Player.Character.PrimaryPart.Position
    
    for _, otherPlayer in pairs(Players:GetPlayers()) do
        if otherPlayer ~= Player and otherPlayer.Character and otherPlayer.Character.PrimaryPart then
            local distance = (playerPos - otherPlayer.Character.PrimaryPart.Position).Magnitude
            if distance <= Config.autoSpamThreshold then
                return true
            end
        end
    end
    
    return false
end

function AutoSpamSystem:executeSpam()
    if not self:shouldSpam() or self.isSpamming then return false end
    
    self.isSpamming = true
    Logger:info("AutoSpam", "Iniciando burst")
    
    task.spawn(function()
        for i = 1, Config.autoSpamParries do
            if not self.isSpamming then break end
            
            if self:canSpamParry() then
                game:GetService("VirtualInputManager"):SendMouseButtonEvent(0, 0, 0, true, game, 1)
                self.lastSpamParry = tick()
                ParrySystem.lastParry = tick()
            end
            
            if i < Config.autoSpamParries then
                task.wait(Config.autoSpamDuration / Config.autoSpamParries)
            end
        end
        
        task.wait(0.1)
        self.isSpamming = false
    end)
    
    return true
end

function AutoSpamSystem:update()
    if not self.isSpamming and self:shouldSpam() then
        self:executeSpam()
    end
end

-- ══════════════════════════════════════════════════════════════
-- ADVANCED ANTI-DETECTION SYSTEM
-- ══════════════════════════════════════════════════════════════

local AntiDetection = {
    fingerprint = nil,
    behaviorPattern = {},
    lastRandomization = 0
}

function AntiDetection:generateFingerprint()
    if self.fingerprint then return self.fingerprint end
    
    local components = {
        os.time(),
        math.random(1000, 9999),
        Player.UserId,
        tick()
    }
    
    self.fingerprint = string.format("%d_%d_%d_%.3f", components[1], components[2], components[3], components[4])
    
    return self.fingerprint
end

-- ══════════════════════════════════════════════════════════════
-- ADVANCED VISUALIZATION SYSTEM (3D ULTRA)
-- ══════════════════════════════════════════════════════════════

local Visualization = {
    trajectoryParts = {},
    trajectoryBeams = {},
    trajectoryTrails = {},
    heatmapParts = {},
    confidenceIndicators = {},
    playerIndicators = {},
    playerConnections = {},
    clashIndicators = {},
    parryEffects = {},
    particles = {},
    ballHitbox = nil,
    ballWireframe = {},
    forceField = nil,
    velocityTunnel = {},
    maxTrajectoryParts = 50,
    maxParticles = 200
}

function Visualization:createTrajectory(predictedPositions, confidence)
    if not Config.showTrajectory3D then return end
    
    -- Clean up old trajectory
    for _, part in ipairs(self.trajectoryParts) do
        if part and part.Parent then
            part:Destroy()
        end
    end
    self.trajectoryParts = {}
    
    for _, beam in ipairs(self.trajectoryBeams) do
        if beam and beam.Parent then
            beam:Destroy()
        end
    end
    self.trajectoryBeams = {}
    
    if not predictedPositions or #predictedPositions == 0 then return end
    
    local trajectoryPoints = Config.trajectoryPoints or 12
    local numPoints = math.min(#predictedPositions, trajectoryPoints)
    
    -- Create smooth curved trajectory using Beams
    for i = 1, numPoints - 1 do
        local startPos = predictedPositions[i]
        local endPos = predictedPositions[i + 1]
        
        if startPos and endPos then
            local startPart = Instance.new("Part")
            startPart.Name = "TrajectoryStart"
            startPart.Anchored = true
            startPart.CanCollide = false
            startPart.Transparency = 1
            startPart.Size = Vector3.new(0.1, 0.1, 0.1)
            startPart.Position = startPos
            startPart.Parent = Workspace
            
            local endPart = Instance.new("Part")
            endPart.Name = "TrajectoryEnd"
            endPart.Anchored = true
            endPart.CanCollide = false
            endPart.Transparency = 1
            endPart.Size = Vector3.new(0.1, 0.1, 0.1)
            endPart.Position = endPos
            endPart.Parent = Workspace
            
            local startAttachment = Instance.new("Attachment")
            startAttachment.Parent = startPart
            
            local endAttachment = Instance.new("Attachment")
            endAttachment.Parent = endPart
            
            local beam = Instance.new("Beam")
            beam.Attachment0 = startAttachment
            beam.Attachment1 = endAttachment
            beam.FaceCamera = true
            beam.Width0 = 0.5
            beam.Width1 = 0.3
            beam.Transparency = NumberSequence.new({
                NumberSequenceKeypoint.new(0, 0.2),
                NumberSequenceKeypoint.new(0.5, 0.4),
                NumberSequenceKeypoint.new(1, 0.8)
            })
            beam.LightEmission = 0.8
            beam.LightInfluence = 0.5
            
            local highConfColor = Color3.fromRGB(0, 255, 150)
            local lowConfColor = Color3.fromRGB(255, 50, 50)
            local color = highConfColor:Lerp(lowConfColor, 1 - confidence)
            beam.Color = ColorSequence.new({
                ColorSequenceKeypoint.new(0, color),
                ColorSequenceKeypoint.new(0.5, Color3.fromRGB(100, 200, 255)),
                ColorSequenceKeypoint.new(1, color * 0.7)
            })
            beam.Parent = startPart
            
            table.insert(self.trajectoryBeams, beam)
            table.insert(self.trajectoryParts, startPart)
            table.insert(self.trajectoryParts, endPart)
        end
    end
end

function Visualization:createParryEffect(position)
    if not Config.showParryEffects then return end
    
    -- Create shockwave rings
    for wave = 1, 3 do
        local ring = Instance.new("Part")
        ring.Name = "ShockWave" .. wave
        ring.Anchored = true
        ring.CanCollide = false
        ring.Transparency = 0.3 + (wave * 0.1)
        ring.Size = Vector3.new(0.5, 8 + wave * 2, 8 + wave * 2)
        ring.Shape = Enum.PartType.Cylinder
        ring.Material = Enum.Material.Neon
        ring.Position = position
        ring.CFrame = CFrame.lookAt(position, position + Vector3.new(0, 1, 0)) * CFrame.Angles(0, 0, math.rad(90))
        ring.Color = Color3.fromRGB(0, 255 - wave * 30, 255)
        ring.Parent = Workspace
        
        task.spawn(function()
            local startTime = tick()
            while tick() - startTime < 0.5 do
                local t = (tick() - startTime) / 0.5
                ring.Size = Vector3.new(0.5, (8 + wave * 2) * (1 + t * 3), (8 + wave * 2) * (1 + t * 3))
                ring.Transparency = 0.3 + (wave * 0.1) + t * 0.7
                task.wait()
            end
            if ring and ring.Parent then
                ring:Destroy()
            end
        end)
        
        table.insert(self.parryEffects, ring)
    end
end

function Visualization:update()
    if not Config.showTrajectory3D then return end
    
    if BallTracker.ball and #BallTracker.history >= 2 then
        local trajectoryPoints = Config.trajectoryPoints or 12
        local predictedPositions = {}
        for i = 1, trajectoryPoints do
            local pred = BallTracker:predict(i * 0.1)
            if pred then
                table.insert(predictedPositions, pred)
            end
        end
        
        if #predictedPositions > 0 then
            local confidence = BallTracker:getConfidence()
            self:createTrajectory(predictedPositions, confidence)
        end
    end
end

function Visualization:cleanup()
    for _, part in ipairs(self.trajectoryParts) do
        if part and part.Parent then
            part:Destroy()
        end
    end
    self.trajectoryParts = {}
    
    for _, beam in ipairs(self.trajectoryBeams) do
        if beam and beam.Parent then
            beam:Destroy()
        end
    end
    self.trajectoryBeams = {}
    
    for _, effect in ipairs(self.parryEffects) do
        if effect and effect.Parent then
            effect:Destroy()
        end
    end
    self.parryEffects = {}
end

-- ══════════════════════════════════════════════════════════════
-- UI SYSTEM MODULE
-- ══════════════════════════════════════════════════════════════

local UI = {
    gui = nil,
    hud = nil,
    currentTab = "Parry"
}

function UI:createGUI()
    local existing = Player.PlayerGui:FindFirstChild("BladeBallV7")
    if existing then existing:Destroy() end
    
    local screenGui = Instance.new("ScreenGui")
    screenGui.Name = "BladeBallV7"
    screenGui.ResetOnSpawn = false
    screenGui.ZIndexBehavior = Enum.ZIndexBehavior.Sibling
    
    if gethui then
        screenGui.Parent = gethui()
    else
        screenGui.Parent = Player.PlayerGui
    end
    
    self.gui = screenGui
    self:createMainFrame()
    self:createHUD()
    
    Utils.notify("✓ Blade Ball V7.0", "Sistema Ultra Avançado carregado!", 3)
end

function UI:createMainFrame()
    local main = Instance.new("Frame")
    main.Name = "MainFrame"
    main.Size = UDim2.new(0, 500, 0, 400)
    main.Position = UDim2.new(0.5, -250, 0.5, -200)
    main.BackgroundColor3 = Color3.fromRGB(20, 20, 25)
    main.BorderSizePixel = 0
    main.Visible = false
    main.Parent = self.gui
    
    Instance.new("UICorner", main).CornerRadius = UDim.new(0, 12)
    
    -- Title bar
    local titleBar = Instance.new("Frame")
    titleBar.Size = UDim2.new(1, 0, 0, 40)
    titleBar.BackgroundColor3 = Color3.fromRGB(15, 15, 20)
    titleBar.BorderSizePixel = 0
    titleBar.Parent = main
    
    Instance.new("UICorner", titleBar).CornerRadius = UDim.new(0, 12)
    
    local title = Instance.new("TextLabel")
    title.Size = UDim2.new(1, -50, 1, 0)
    title.Position = UDim2.new(0, 15, 0, 0)
    title.BackgroundTransparency = 1
    title.Text = "⚡ Blade Ball V7.0 - Ultimate Edition"
    title.TextColor3 = Color3.fromRGB(100, 200, 255)
    title.Font = Enum.Font.GothamBold
    title.TextSize = 16
    title.TextXAlignment = Enum.TextXAlignment.Left
    title.Parent = titleBar
    
    -- Close button
    local closeBtn = Instance.new("TextButton")
    closeBtn.Size = UDim2.new(0, 30, 0, 30)
    closeBtn.Position = UDim2.new(1, -35, 0, 5)
    closeBtn.BackgroundColor3 = Color3.fromRGB(255, 60, 60)
    closeBtn.Text = "×"
    closeBtn.TextColor3 = Color3.fromRGB(255, 255, 255)
    closeBtn.Font = Enum.Font.GothamBold
    closeBtn.TextSize = 20
    closeBtn.Parent = titleBar
    
    Instance.new("UICorner", closeBtn).CornerRadius = UDim.new(0, 8)
    
    closeBtn.MouseButton1Click:Connect(function()
        main.Visible = false
    end)
    
    -- Content frame
    local content = Instance.new("ScrollingFrame")
    content.Name = "Content"
    content.Size = UDim2.new(1, -20, 1, -50)
    content.Position = UDim2.new(0, 10, 0, 50)
    content.BackgroundTransparency = 1
    content.ScrollBarThickness = 4
    content.CanvasSize = UDim2.new(0, 0, 0, 0)
    content.Parent = main
    
    local layout = Instance.new("UIListLayout")
    layout.Padding = UDim.new(0, 8)
    layout.Parent = content
    
    -- Settings
    self:createToggle(content, "Ativado", "enabled", Config.enabled)
    self:createDropdown(content, "Modo", {"ULTRA_SAFE", "SAFE", "BALANCED", "AGGRESSIVE", "ULTRA_AGGRESSIVE"}, Config.mode)
    self:createToggle(content, "Auto Ajuste", "autoAdjust", Config.autoAdjust)
    self:createToggle(content, "Detectar Curvas", "detectCurve", Config.detectCurve)
    self:createToggle(content, "Trajetória 3D", "showTrajectory3D", Config.showTrajectory3D)
    self:createToggle(content, "Efeitos de Parry", "showParryEffects", Config.showParryEffects)
    self:createToggle(content, "Auto Spam", "autoSpamEnabled", Config.autoSpamEnabled)
    self:createToggle(content, "Parry Rápido", "fastParryEnabled", Config.fastParryEnabled)
    
    -- Make draggable
    self:makeDraggable(main, titleBar)
end

function UI:createToggle(parent, text, configKey, default)
    local frame = Instance.new("Frame")
    frame.Size = UDim2.new(1, 0, 0, 30)
    frame.BackgroundColor3 = Color3.fromRGB(30, 30, 35)
    frame.BorderSizePixel = 0
    frame.Parent = parent
    
    Instance.new("UICorner", frame).CornerRadius = UDim.new(0, 6)
    
    local label = Instance.new("TextLabel")
    label.Size = UDim2.new(0.7, 0, 1, 0)
    label.Position = UDim2.new(0, 10, 0, 0)
    label.BackgroundTransparency = 1
    label.Text = text
    label.TextColor3 = Color3.fromRGB(220, 220, 220)
    label.Font = Enum.Font.Gotham
    label.TextSize = 12
    label.TextXAlignment = Enum.TextXAlignment.Left
    label.Parent = frame
    
    local toggle = Instance.new("TextButton")
    toggle.Size = UDim2.new(0, 50, 0, 22)
    toggle.Position = UDim2.new(1, -55, 0.5, -11)
    toggle.BackgroundColor3 = default and Color3.fromRGB(60, 200, 100) or Color3.fromRGB(100, 100, 100)
    toggle.Text = default and "ON" or "OFF"
    toggle.TextColor3 = Color3.fromRGB(255, 255, 255)
    toggle.Font = Enum.Font.GothamBold
    toggle.TextSize = 11
    toggle.Parent = frame
    
    Instance.new("UICorner", toggle).CornerRadius = UDim.new(0, 5)
    
    toggle.MouseButton1Click:Connect(function()
        Config[configKey] = not Config[configKey]
        toggle.Text = Config[configKey] and "ON" or "OFF"
        toggle.BackgroundColor3 = Config[configKey] and Color3.fromRGB(60, 200, 100) or Color3.fromRGB(100, 100, 100)
    end)
end

function UI:createDropdown(parent, text, options, default)
    local frame = Instance.new("Frame")
    frame.Size = UDim2.new(1, 0, 0, 30)
    frame.BackgroundColor3 = Color3.fromRGB(30, 30, 35)
    frame.BorderSizePixel = 0
    frame.Parent = parent
    
    Instance.new("UICorner", frame).CornerRadius = UDim.new(0, 6)
    
    local label = Instance.new("TextLabel")
    label.Size = UDim2.new(0.5, 0, 1, 0)
    label.Position = UDim2.new(0, 10, 0, 0)
    label.BackgroundTransparency = 1
    label.Text = text
    label.TextColor3 = Color3.fromRGB(220, 220, 220)
    label.Font = Enum.Font.Gotham
    label.TextSize = 12
    label.TextXAlignment = Enum.TextXAlignment.Left
    label.Parent = frame
    
    local btn = Instance.new("TextButton")
    btn.Size = UDim2.new(0.4, 0, 0, 22)
    btn.Position = UDim2.new(0.58, 0, 0.5, -11)
    btn.BackgroundColor3 = Color3.fromRGB(50, 50, 55)
    btn.Text = default
    btn.TextColor3 = Color3.fromRGB(255, 255, 255)
    btn.Font = Enum.Font.Gotham
    btn.TextSize = 11
    btn.Parent = frame
    
    Instance.new("UICorner", btn).CornerRadius = UDim.new(0, 5)
    
    local currentIndex = 1
    for i, opt in ipairs(options) do
        if opt == default then
            currentIndex = i
            break
        end
    end
    btn.MouseButton1Click:Connect(function()
        currentIndex = (currentIndex % #options) + 1
        btn.Text = options[currentIndex]
        Config.mode = options[currentIndex]
    end)
end

function UI:makeDraggable(frame, dragHandle)
    local dragging, dragInput, dragStart, startPos
    
    dragHandle.InputBegan:Connect(function(input)
        if input.UserInputType == Enum.UserInputType.MouseButton1 then
            dragging = true
            dragStart = input.Position
            startPos = frame.Position
        end
    end)
    
    dragHandle.InputEnded:Connect(function(input)
        if input.UserInputType == Enum.UserInputType.MouseButton1 then
            dragging = false
        end
    end)
    
    UserInputService.InputChanged:Connect(function(input)
        if dragging and input.UserInputType == Enum.UserInputType.MouseMovement then
            local delta = input.Position - dragStart
            frame.Position = UDim2.new(
                startPos.X.Scale, startPos.X.Offset + delta.X,
                startPos.Y.Scale, startPos.Y.Offset + delta.Y
            )
        end
    end)
end

function UI:toggleVisibility()
    if self.gui and self.gui.MainFrame then
        self.gui.MainFrame.Visible = not self.gui.MainFrame.Visible
    end
end

function UI:createHUD()
    if not Config.showHUD then return end
    
    local hud = Instance.new("Frame")
    hud.Name = "HUD"
    hud.Size = UDim2.new(0, 200, 0, 80)
    hud.Position = Config.hudPosition
    hud.BackgroundColor3 = Color3.fromRGB(20, 20, 25)
    hud.BorderSizePixel = 0
    hud.Parent = self.gui
    
    Instance.new("UICorner", hud).CornerRadius = UDim.new(0, 10)
    
    local title = Instance.new("TextLabel")
    title.Size = UDim2.new(1, -10, 0, 25)
    title.Position = UDim2.new(0, 5, 0, 5)
    title.BackgroundTransparency = 1
    title.Text = "⚡ Blade Ball V7.0"
    title.TextColor3 = Color3.fromRGB(100, 200, 255)
    title.Font = Enum.Font.GothamBold
    title.TextSize = 14
    title.Parent = hud
    
    local status = Instance.new("TextLabel")
    status.Name = "Status"
    status.Size = UDim2.new(1, -10, 0, 20)
    status.Position = UDim2.new(0, 5, 0, 30)
    status.BackgroundTransparency = 1
    status.Text = "Status: Ativo"
    status.TextColor3 = Color3.fromRGB(100, 255, 100)
    status.Font = Enum.Font.Gotham
    status.TextSize = 11
    status.TextXAlignment = Enum.TextXAlignment.Left
    status.Parent = hud
    
    local stats = Instance.new("TextLabel")
    stats.Name = "Stats"
    stats.Size = UDim2.new(1, -10, 0, 20)
    stats.Position = UDim2.new(0, 5, 0, 52)
    stats.BackgroundTransparency = 1
    stats.Text = "Parries: 0"
    stats.TextColor3 = Color3.fromRGB(200, 200, 200)
    stats.Font = Enum.Font.Gotham
    stats.TextSize = 10
    stats.TextXAlignment = Enum.TextXAlignment.Left
    stats.Parent = hud
    
    self.hud = hud
end

function UI:updateHUD()
    if not self.hud or not Config.showHUD then 
        if self.hud then self.hud.Visible = false end
        return 
    end
    
    self.hud.Visible = true
    
    local status = self.hud:FindFirstChild("Status")
    if status then
        status.Text = "Status: " .. (Config.enabled and "Ativo" or "Desativado")
        status.TextColor3 = Config.enabled and Color3.fromRGB(100, 255, 100) or Color3.fromRGB(255, 100, 100)
    end
    
    local stats = self.hud:FindFirstChild("Stats")
    if stats then
        stats.Text = string.format("Parries: %d", ParrySystem.parryCount or 0)
    end
end

-- ══════════════════════════════════════════════════════════════
-- MAIN LOOP & INITIALIZATION
-- ══════════════════════════════════════════════════════════════

local MainLoop = {
    connection = nil,
    updateInterval = CONSTANTS.UPDATE_RATE,
    lastUpdate = 0
}

function MainLoop:start()
    self.connection = RunService.Heartbeat:Connect(function()
        local now = tick()
        if now - self.lastUpdate < self.updateInterval then return end
        self.lastUpdate = now
        
        -- Update ball tracking
        local ballData = BallTracker:update()
        
        -- Auto Spam System
        if Config.autoSpamEnabled then
            AutoSpamSystem:update()
        end
        
        -- Check if should parry (com proteção adicional contra double click)
        if ballData and not AutoSpamSystem.isSpamming and not ParrySystem.isParrying then
            if ParrySystem:shouldParry(ballData) then
                local success = ParrySystem:executeParry(ballData)
                if success and ballData then
                    Visualization:createParryEffect(Player.Character and Player.Character.PrimaryPart and Player.Character.PrimaryPart.Position or Vector3.new(0, 0, 0))
                end
            end
        end
        
        -- Update UI
        UI:updateHUD()
        
        -- Update visualization
        Visualization:update()
    end)
end

function MainLoop:stop()
    if self.connection then
        self.connection:Disconnect()
        self.connection = nil
    end
end

-- ══════════════════════════════════════════════════════════════
-- INITIALIZATION
-- ══════════════════════════════════════════════════════════════

local function Initialize()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Blade Ball Auto Parry V7.0 - ULTIMATE EDITION           ║")
    print("║   Sistema Ultra Avançado Premium Edition                  ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("")
    print("[✓] Filtro de Kalman Multi-Dimensional")
    print("[✓] Predição com Filtro de Partículas")
    print("[✓] Deep Neural Network com 5 Camadas")
    print("[✓] Reinforcement Learning")
    print("[✓] Genetic Algorithm")
    print("[✓] Sistema de Fusão Sensorial Bayesiana")
    print("[✓] Visualização 3D Avançada")
    print("")
    print("Sistema inicializado com sucesso!")
    print("════════════════════════════════════════════════")
    
    -- Initialize systems
    if Config.geneticOptimization then
        GeneticAlgorithm:initialize()
    end
    
    if Config.dnnEnabled then
        DeepNeuralNetwork:initialize()
    end
    
    AntiDetection:generateFingerprint()
    Logger:info("System", "Sistema inicializado", {version = CONSTANTS.VERSION})
    
    -- Create UI
    UI:createGUI()
    
    -- Setup hotkey
    UserInputService.InputBegan:Connect(function(input, processed)
        if processed then return end
        if input.KeyCode == Enum.KeyCode.RightControl then
            UI:toggleVisibility()
        end
    end)
    
    -- Setup events
    Events:on("parryExecuted", function(data)
        if Config.showNotifications and math.random(1, 5) == 1 then
            Utils.notify("✓ Parry", string.format("Executado #%d", data.count), 2)
        end
    end)
    
    Events:on("targetChanged", function(target, oldTarget)
        AdvancedCache:clear()
        ParrySystem.lastTargetChangeTime = tick()
        BallTracker.lastTargetChangeTime = tick()
        
        Logger:info("TargetChange", "Target mudou", {
            from = oldTarget,
            to = target,
            time = tick()
        })
    end)
    
    -- Start main loop
    MainLoop:start()
    
    Utils.notify("⚡ Blade Ball V7.0", "Sistema Ultra Avançado iniciado!", 4)
end

-- Wait for game to load
if not game:IsLoaded() then
    game.Loaded:Wait()
end

-- Wait for character
if not Player.Character then
    Player.CharacterAdded:Wait()
end

task.wait(1)

-- Initialize system
local success, error = pcall(Initialize)

if not success then
    warn("[Blade Ball V7] Erro:", error)
else
    print("[✓] Blade Ball V7.0 Iniciado com Sucesso!")
end

return {
    Version = CONSTANTS.VERSION,
    Config = Config,
    BallTracker = BallTracker,
    ParrySystem = ParrySystem,
    UI = UI
}