    // frontend/src/services/api.js - Conexión al backend Flask

const API_URL = 'http://localhost:5000/api';

export const analyzeText = async (text) => {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });
        
        const data = await response.json();
        
        if (data.success) {
            return {
                veredicto: data.veredicto,
                confidence: data.confidence,
                confidencePercent: data.confidence_percent,
                topFeatures: data.top_features,
                message: data.message,
                isIA: data.is_ia
            };
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Error al analizar:', error);
        throw error;
    }
};

// Verificar estado del servidor
export const checkHealth = async () => {
    const response = await fetch(`${API_URL}/health`);
    return response.json();
};