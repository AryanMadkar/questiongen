from flask import Flask, request, jsonify, render_template_string
from groq import Groq
import os
import json
import hashlib
import time
import logging
from datetime import datetime, timedelta
from functools import wraps
import re
from typing import Dict, List, Optional, Tuple
import threading
from groq import Groq
from collections import defaultdict
from dotenv import load_dotenv
import httpx




load_dotenv()  # Loads the .env file variables into environment variables
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
httpx_client = httpx.Client(proxies=None)

app = Flask(__name__)
api_key = os.getenv('API_KEY')
json_sort_key = os.getenv("JSON_SORT_KEYS")

app.config[json_sort_key] = False
# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", api_key)
client = Groq(api_key=GROQ_API_KEY, http_client=httpx_client)

# Enhanced caching system
class AdvancedCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.creation_times[key] > self.ttl_seconds:
                    self._remove(key)
                    return None
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.creation_times[key] = time.time()
    
    def _remove(self, key):
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.creation_times[key]
    
    def _evict_lru(self):
        if not self.cache:
            return
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(lru_key)

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier):
        with self.lock:
            now = time.time()
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.window_seconds
            ]
            
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(now)
                return True
            return False

# Initialize systems
cache = AdvancedCache(max_size=2000, ttl_seconds=7200)  # 2 hour TTL
rate_limiter = RateLimiter(max_requests=50, window_seconds=3600)  # 50 requests per hour

# Enhanced prompt templates
PROMPT_TEMPLATES = {
    "academic": """
Generate {num_questions} academically rigorous multiple-choice questions about "{topic}" at {difficulty} level.

Requirements:
- Questions must test deep understanding, not just memorization
- Include application, analysis, and synthesis level questions
- Options should be plausible and challenging
- Avoid obvious incorrect answers
- Include explanations for correct answers

Difficulty Guidelines:
- Easy: Basic concepts and definitions
- Medium: Application and analysis
- Hard: Synthesis, evaluation, and complex problem-solving
- Expert: Advanced theoretical concepts and real-world applications

Return ONLY this JSON structure:
{{
  "metadata": {{
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "total_questions": {num_questions},
    "generation_time": "{timestamp}",
    "bloom_taxonomy_levels": ["remember", "understand", "apply", "analyze", "evaluate", "create"]
  }},
  "questions": [
    {{
      "id": 1,
      "question": "Clear, specific question text",
      "options": {{
        "A": "First option",
        "B": "Second option", 
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct_answer": "A",
      "explanation": "Detailed explanation of why this answer is correct",
      "bloom_level": "analyze",
      "estimated_time_seconds": 45,
      "tags": ["concept1", "concept2"]
    }}
  ]
}}
""",
    
    "practical": """
Generate {num_questions} practical, scenario-based multiple-choice questions about "{topic}" at {difficulty} level.

Focus on:
- Real-world applications and case studies
- Problem-solving scenarios
- Best practices and common mistakes
- Industry standards and procedures

Return the same JSON structure as academic template.
""",
    
    "conceptual": """
Generate {num_questions} conceptual multiple-choice questions about "{topic}" at {difficulty} level.

Focus on:
- Theoretical understanding
- Relationships between concepts
- Cause and effect relationships
- Comparative analysis

Return the same JSON structure as academic template.
"""
}

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": 3600
            }), 429
        return f(*args, **kwargs)
    return decorated_function

def validate_input(data: Dict) -> Tuple[bool, str]:
    """Enhanced input validation"""
    if not data:
        return False, "No data provided"
    
    topic = data.get("topic", "").strip()
    if not topic or len(topic) < 2:
        return False, "Topic must be at least 2 characters long"
    
    if len(topic) > 200:
        return False, "Topic must be less than 200 characters"
    
    difficulty = data.get("difficulty", "medium").lower()
    valid_difficulties = ["easy", "medium", "hard", "expert"]
    if difficulty not in valid_difficulties:
        return False, f"Difficulty must be one of: {', '.join(valid_difficulties)}"
    
    num_questions = data.get("num_questions", 5)
    if not isinstance(num_questions, int) or num_questions < 1 or num_questions > 50:
        return False, "Number of questions must be between 1 and 50"
    
    question_type = data.get("question_type", "academic")
    valid_types = ["academic", "practical", "conceptual"]
    if question_type not in valid_types:
        return False, f"Question type must be one of: {', '.join(valid_types)}"
    
    return True, ""

def generate_cache_key(topic: str, difficulty: str, num_questions: int, question_type: str) -> str:
    """Generate a secure cache key"""
    content = f"{topic.lower()}_{difficulty.lower()}_{num_questions}_{question_type}"
    return hashlib.md5(content.encode()).hexdigest()

def build_enhanced_prompt(topic: str, difficulty: str, num_questions: int, question_type: str) -> str:
    """Build enhanced prompt based on question type"""
    timestamp = datetime.now().isoformat()
    template = PROMPT_TEMPLATES.get(question_type, PROMPT_TEMPLATES["academic"])
    
    return template.format(
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
        timestamp=timestamp
    )

def validate_generated_content(content: str) -> Tuple[bool, Optional[Dict]]:
    """Validate and parse generated content"""
    try:
        # Find JSON in response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return False, None
        
        json_data = json.loads(json_match.group())
        
        # Validate structure
        required_fields = ["metadata", "questions"]
        if not all(field in json_data for field in required_fields):
            return False, None
        
        questions = json_data.get("questions", [])
        if not questions:
            return False, None
        
        # Validate each question
        for i, q in enumerate(questions):
            required_q_fields = ["question", "options", "correct_answer"]
            if not all(field in q for field in required_q_fields):
                logger.warning(f"Question {i+1} missing required fields")
                return False, None
            
            # Validate options
            options = q.get("options", {})
            if len(options) != 4 or not all(key in options for key in ["A", "B", "C", "D"]):
                logger.warning(f"Question {i+1} has invalid options structure")
                return False, None
            
            # Validate correct answer
            if q.get("correct_answer") not in ["A", "B", "C", "D"]:
                logger.warning(f"Question {i+1} has invalid correct answer")
                return False, None
        
        return True, json_data
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Content validation error: {e}")
        return False, None

def enhance_response(json_data: Dict) -> Dict:
    """Add enhancements to the response"""
    # Add quality metrics
    questions = json_data.get("questions", [])
    
    # Calculate estimated total time
    total_time = sum(q.get("estimated_time_seconds", 60) for q in questions)
    
    # Add statistics
    bloom_levels = [q.get("bloom_level", "remember") for q in questions]
    bloom_distribution = {level: bloom_levels.count(level) for level in set(bloom_levels)}
    
    json_data["analytics"] = {
        "total_estimated_time_minutes": round(total_time / 60, 1),
        "average_time_per_question": round(total_time / len(questions), 1),
        "bloom_taxonomy_distribution": bloom_distribution,
        "difficulty_score": calculate_difficulty_score(json_data.get("metadata", {}).get("difficulty", "medium")),
        "quality_indicators": {
            "has_explanations": all("explanation" in q for q in questions),
            "has_bloom_levels": all("bloom_level" in q for q in questions),
            "has_tags": all("tags" in q for q in questions)
        }
    }
    
    return json_data

def calculate_difficulty_score(difficulty: str) -> float:
    """Calculate numerical difficulty score"""
    scores = {"easy": 0.25, "medium": 0.5, "hard": 0.75, "expert": 1.0}
    return scores.get(difficulty.lower(), 0.5)

@app.route('/')
def home():
    """API documentation page"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced MCQ Generator API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { background: #28a745; color: white; padding: 4px 8px; border-radius: 3px; font-size: 12px; }
            code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
            .example { background: #f1f3f4; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Enhanced MCQ Generator API</h1>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /generate_mcqs</h3>
                <p><strong>Generate high-quality multiple choice questions</strong></p>
                
                <h4>Request Body:</h4>
                <div class="example">
                    <code>
                    {<br>
                    &nbsp;&nbsp;"topic": "Machine Learning",<br>
                    &nbsp;&nbsp;"difficulty": "medium",<br>
                    &nbsp;&nbsp;"num_questions": 5,<br>
                    &nbsp;&nbsp;"question_type": "academic"<br>
                    }
                    </code>
                </div>
                
                <h4>Parameters:</h4>
                <ul>
                    <li><code>topic</code> (required): Subject matter (2-200 characters)</li>
                    <li><code>difficulty</code>: easy, medium, hard, expert (default: medium)</li>
                    <li><code>num_questions</code>: 1-50 questions (default: 5)</li>
                    <li><code>question_type</code>: academic, practical, conceptual (default: academic)</li>
                </ul>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /health</h3>
                <p>Check API health status</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /stats</h3>
                <p>Get API usage statistics</p>
            </div>
            
            <h3>🚀 Features:</h3>
            <ul>
                <li>Advanced caching with TTL</li>
                <li>Rate limiting protection</li>
                <li>Multiple question types</li>
                <li>Bloom's taxonomy classification</li>
                <li>Quality validation</li>
                <li>Detailed explanations</li>
                <li>Analytics and metrics</li>
            </ul>
        </div>
    </body>
    </html>
    """)

@app.route('/generate_mcqs', methods=['POST'])
@rate_limit
def generate_mcqs():
    """Enhanced MCQ generation endpoint"""
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({"error": "Validation failed", "message": error_msg}), 400
        
        # Extract parameters
        topic = data["topic"].strip()
        difficulty = data.get("difficulty", "medium").lower()
        num_questions = data.get("num_questions", 5)
        question_type = data.get("question_type", "academic")
        
        # Check cache
        cache_key = generate_cache_key(topic, difficulty, num_questions, question_type)
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for topic: {topic}")
            return jsonify({**cached_result, "cached": True})
        
        # Build prompt
        prompt = build_enhanced_prompt(topic, difficulty, num_questions, question_type)
        
        # Generate with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating MCQs (attempt {attempt + 1}) - Topic: {topic}, Difficulty: {difficulty}")
                
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert educational content creator specializing in high-quality multiple-choice questions. Always respond with valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,  # Lower temperature for more consistent results
                    max_tokens=4000,
                    top_p=0.9
                )
                
                content = response.choices[0].message.content.strip()
                
                # Validate generated content
                is_valid_content, json_data = validate_generated_content(content)
                if is_valid_content and json_data:
                    # Enhance response
                    enhanced_data = enhance_response(json_data)
                    
                    # Cache the result
                    cache.set(cache_key, enhanced_data)
                    
                    logger.info(f"Successfully generated {len(enhanced_data.get('questions', []))} questions for topic: {topic}")
                    return jsonify({**enhanced_data, "cached": False})
                
                logger.warning(f"Invalid content generated on attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # Brief pause before retry
        
        return jsonify({
            "error": "Generation failed",
            "message": "Unable to generate valid questions after multiple attempts"
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_mcqs: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "cache_size": len(cache.cache),
        "uptime": "Available"
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API statistics"""
    return jsonify({
        "cache_stats": {
            "current_size": len(cache.cache),
            "max_size": cache.max_size,
            "ttl_seconds": cache.ttl_seconds
        },
        "rate_limit_stats": {
            "max_requests_per_hour": rate_limiter.max_requests,
            "window_seconds": rate_limiter.window_seconds
        },
        "supported_features": {
            "question_types": ["academic", "practical", "conceptual"],
            "difficulty_levels": ["easy", "medium", "hard", "expert"],
            "max_questions": 50,
            "bloom_taxonomy": True,
            "explanations": True,
            "analytics": True
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting Enhanced MCQ Generator API...")
    logger.info(f"Cache configured: max_size={cache.max_size}, ttl={cache.ttl_seconds}s")
    logger.info(f"Rate limiting: {rate_limiter.max_requests} requests per hour")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
