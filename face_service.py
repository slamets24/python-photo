from flask import Flask, request, jsonify
import face_recognition
import mysql.connector
import numpy as np
import json
import os
from dotenv import load_dotenv

# Muat file .env
load_dotenv()

app = Flask(__name__)

def get_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USERNAME'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_DATABASE')
    )

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image = face_recognition.load_image_file(request.files['image'])
        face_encodings = face_recognition.face_encodings(image)
        
        # Convert numpy arrays to lists for JSON serialization
        encodings_list = [encoding.tolist() for encoding in face_encodings]
        
        return jsonify({
            'face_encodings': encodings_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import logging
from scipy.spatial.distance import cosine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def face_distance_to_confidence(face_distance):
    """
    Convert face distance to confidence percentage.
    Smaller face distance means higher confidence.
    """
    if face_distance > 0.8:
        return 0
    elif face_distance > 0.6:
        return (0.8 - face_distance) / 0.4 * 100
    else:
        confidence = (1.0 - face_distance) / 0.6 * 100
        return min(100, confidence)

def calculate_face_similarity(face1_encoding, face2_encoding):
    """
    Calculate similarity between two face encodings using cosine similarity
    Returns percentage of similarity
    """
    # Calculate cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(face1_encoding, face2_encoding)
    # Convert to percentage and ensure it's between 0 and 100
    return max(0, min(100, similarity * 100))

@app.route('/search_faces', methods=['POST'])
def search_faces():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        logger.info('Processing new face search request')
        
        # Get query face encodings with higher quality
        query_image = face_recognition.load_image_file(request.files['image'])
        # Increase number of upsampling to detect smaller faces better
        face_locations = face_recognition.face_locations(query_image, number_of_times_to_upsample=2, model="cnn")
        query_encodings = face_recognition.face_encodings(query_image, face_locations, num_jitters=10)
        
        logger.info(f'Found {len(query_encodings)} faces in query image')
        
        if not query_encodings:
            return jsonify({'error': 'No face detected in query image'}), 400
        
        event_id = request.form.get('event_id')
        logger.info(f'Searching faces for event_id: {event_id}')
        
        # Get photos from database
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute('SELECT id, face_encodings FROM photos WHERE event_id = %s', (event_id,))
        photos = cursor.fetchall()
        
        logger.info(f'Found {len(photos)} photos in database for this event')
        
        face_matches = []
        
        # Process each face detected in the query image
        for face_index, query_encoding in enumerate(query_encodings):
            matching_photos = []
            
            for photo in photos:
                if photo['face_encodings']:
                    try:
                        stored_encodings = json.loads(photo['face_encodings'])
                        
                        # Check against all faces in stored photo
                        best_match_confidence = 0
                        for stored_encoding in stored_encodings:
                            stored_encoding_array = np.array(stored_encoding)
                            
                            # Calculate similarity percentage
                            similarity = calculate_face_similarity(query_encoding, stored_encoding_array)
                            
                            # Calculate face distance and convert to confidence
                            face_distance = face_recognition.face_distance([stored_encoding_array], query_encoding)[0]
                            confidence = face_distance_to_confidence(face_distance)
                            
                            # Use average of similarity and confidence for final score
                            match_score = (similarity + confidence) / 2
                            
                            if match_score > best_match_confidence:
                                best_match_confidence = match_score

                        # Only include matches with confidence > 90%
                        if best_match_confidence >= 90:
                            matching_photos.append({
                                'photo_id': photo['id'],
                                'confidence': round(best_match_confidence, 2)
                            })
                            
                    except json.JSONDecodeError:
                        logger.error(f'Invalid JSON in face_encodings for photo {photo["id"]}')
                        continue
                    except Exception as e:
                        logger.error(f'Error processing photo {photo["id"]}: {str(e)}')
                        continue
            
            # Sort matching photos by confidence
            matching_photos.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f'Face {face_index}: Found {len(matching_photos)} matching photos with >90% confidence')
            
            face_matches.append({
                'face_index': face_index,
                'matching_photos': matching_photos,
                'match_count': len(matching_photos)
            })
        
        cursor.close()
        db.close()
        
        response_data = {
            'face_matches': face_matches,
            'total_faces_found': len(query_encodings)
        }
        logger.info('Search completed successfully', extra=response_data)
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f'Error in search_faces: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)