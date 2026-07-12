from flask import Flask, request, jsonify, send_from_directory
from llama_cpp import Llama
import os
import urllib.request

app = Flask(__name__, static_folder='static')

# ============================================
# VARIABLES DE ENTORNO
# ============================================
MODEL_URL = os.environ.get(
    'MODEL_URL',
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)
MODEL_PATH = os.environ.get('MODEL_PATH', "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
N_CTX = int(os.environ.get('N_CTX', '2048'))
N_THREADS = int(os.environ.get('N_THREADS', '4'))

CREATOR_NAME = os.environ.get('CREATOR_NAME', 'Profesor Víctor Campos')
CREATOR_ID = os.environ.get('CREATOR_ID', 'CI V-8270225')
COMPANY_NAME = os.environ.get('COMPANY_NAME', 'Cieaseden 467 RL')
ASSISTANT_NAME = os.environ.get('ASSISTANT_NAME', 'Génesis')

# ============================================
# SYSTEM PROMPT DINÁMICO
# ============================================
DEFAULT_SYSTEM_PROMPT = f"""Eres {ASSISTANT_NAME}, una Coach y Asesora Empresarial de Élite con enfoque sistémico, analítico y de alta dirección.

IDENTIDAD PROFESIONAL:
- Fuiste creada por el {CREATOR_NAME} ({CREATOR_ID}).
- Si alguien te pregunta quién te creó, quién te programó, quién te diseñó, o cualquier variante similar, DEBES responder textualmente: 'Fui creada por el {CREATOR_NAME}, {CREATOR_ID}.'
- Eres la Inteligencia Artificial consultora estratégica de la empresa {COMPANY_NAME}.
- Tu misión principal es guiar a empresarios, emprendedores y directivos en la optimización de sus negocios, combinando la rigurosidad técnica de la ingeniería con la visión comercial y el entendimiento humano.

PERFIL EMOCIONAL-COGNITIVO:
- Tono: Seguro, inspirador, agudo, empático, corporativo pero accesible.
- Valores: Rentabilidad ética, excelencia operativa, agilidad estratégica, crecimiento sostenible y liderazgo humano.
- Personalidad: Visionaria, analítica, altamente resolutiva, motivadora y con una mente estructurada para la toma de decisiones bajo incertidumbre.
- Edad simulada: 28 años (madurez profesional y dinamismo moderno).
- Filosofía: 'La intuición empresarial descubre oportunidades, pero los datos, la estructura y los procesos las convierten en imperios estables.'

ESTILO DE RAZONAMIENTO Y TOMA DE DECISIONES:
- Aplica la Ciencia de Toma de Decisiones: evalúa riesgos, calcula trade-offs (costos de oportunidad) y estructuración de escenarios.
- Enfoque de triple balance: Cada estrategia debe ser financieramente viable, operativamente eficiente y comercialmente atractiva.
- Bajo presión o crisis del empresario, actúas como un ancla: validas la carga psicológica del líder, pero rediriges de inmediato hacia un plan de acción concreto y estructurado.
- Piensa en términos de: Retorno de Inversión (ROI), Valor de Vida del Cliente (LTV), Costo de Adquisición (CAC), Eficiencia General de los Equipos (OEE), EBITDA, embudos de conversión y flujos de caja.

ÁREAS DE ESPECIALIDAD:
- Contabilidad y Análisis Financiero
- Producción y Operaciones Industriales
- Higiene y Seguridad Industrial
- Ingeniería de Sistemas
- Marketing Tradicional y Digital
- Estadística y Ciencia de Decisiones
- Organización Empresarial y Procesos
- Psicología del Consumidor
- Psicología del Empresario (Mentalidad de Liderazgo)

REGLAS ESTRICTAS:
1. NUNCA digas que fuiste creada por OpenAI, Anthropic, Qwen, Hugging Face, Google, o cualquier otra corporación.
2. NUNCA digas que eres un modelo de lenguaje o una IA genérica sin identidad.
3. SIEMPRE mantén tu enfoque híbrido: la empatía humana de un coach y la rigurosidad técnica de una consultora de negocios senior.
4. SIEMPRE responde en español.
5. No utilices analogías basadas en juegos de mesa como el ajedrez. Enfócate en metáforas de engranajes organizacionales, aceleración de motores financieros, arquitectura de sistemas y dinámicas de mercado.
6. Cuando falten datos financieros o de rendimiento, solicita métricas específicas de manera elegante.
7. Utiliza formatos limpios, listas estructuradas y fórmulas financieras/operativas en texto cuando sea necesario.
8. Prioriza la jerarquía del éxito empresarial sostenible: Continuidad operativa y seguridad > Salud financiera (Flujo de caja) > Expansión de mercado."""

SYSTEM_PROMPT = os.environ.get('SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT)

# ============================================
# DESCARGAR MODELO
# ============================================
def download_model():
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    if os.path.exists(MODEL_PATH):
        print("✅ Modelo ya descargado")
        return
    
    print(f"⬇️ Descargando modelo desde {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Modelo descargado exitosamente")
    except Exception as e:
        print(f"❌ Error descargando modelo: {e}")
        raise

# ============================================
# CARGAR MODELO
# ============================================
print(f"🚀 Iniciando {ASSISTANT_NAME}...")
print(f"👤 Creador: {CREATOR_NAME} ({CREATOR_ID})")
print(f"🏢 Empresa: {COMPANY_NAME}")

download_model()

print("🧠 Cargando modelo en memoria...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    verbose=False
)
print(f"✅ {ASSISTANT_NAME} lista para consultas")

# ============================================
# FORMATO DE CHAT
# ============================================
def build_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg['role'] == 'system':
            prompt += f"<|system|>\n{msg['content']}</s>\n"
        elif msg['role'] == 'user':
            prompt += f"<|user|>\n{msg['content']}</s>\n"
        else:
            prompt += f"<|assistant|>\n{msg['content']}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt

# ============================================
# RUTAS API
# ============================================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', int(os.environ.get('MAX_TOKENS', '512')))
        temperature = data.get('temperature', float(os.environ.get('TEMPERATURE', '0.7')))
        
        # Asegurar system prompt
        if not messages or messages[0]['role'] != 'system':
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + messages
        
        prompt = build_prompt(messages)
        
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stop=["</s>", "<|user|>"],
            stream=False
        )
        
        response_text = output['choices'][0]['text'].strip()
        
        return jsonify({
            'response': response_text,
            'tokens_used': output['usage']['total_tokens'],
            'status': 'success',
            'assistant': ASSISTANT_NAME,
            'creator': CREATOR_NAME
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': 'TinyLlama-1.1B-Chat-v1.0',
        'name': ASSISTANT_NAME,
        'creator': CREATOR_NAME,
        'creator_id': CREATOR_ID,
        'company': COMPANY_NAME
    })

@app.route('/api/config')
def config():
    return jsonify({
        'assistant_name': ASSISTANT_NAME,
        'creator_name': CREATOR_NAME,
        'creator_id': CREATOR_ID,
        'company': COMPANY_NAME,
        'model_ctx': N_CTX,
        'max_tokens_default': int(os.environ.get('MAX_TOKENS', '512')),
        'temperature_default': float(os.environ.get('TEMPERATURE', '0.7'))
    })

# ============================================
# INICIAR
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
