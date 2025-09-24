'use client';

import { useState } from 'react';
import styles from './page.module.css';

export default function Home() {
  // --- State Management ---
  // We use 'useState' to store data that can change over time.

  // Stores the actual file object the user uploaded
  const [file, setFile] = useState(null);
  
  // Stores a temporary local URL for the uploaded image, so we can display it immediately
  const [originalImage, setOriginalImage] = useState(null);
  
  // Stores the current value of the epsilon slider
  const [epsilon, setEpsilon] = useState(0.05);
  
  // Stores the JSON response we get back from the FastAPI backend
  const [result, setResult] = useState(null);
  
  // Stores whether we are currently waiting for the API to respond
  const [loading, setLoading] = useState(false);
  
  // Stores any error messages
  const [error, setError] = useState('');

  // --- Functions ---

  // This function is called when the user selects a file
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      // Create a URL for the selected file to show a preview
      setOriginalImage(URL.createObjectURL(selectedFile));
      setResult(null); // Reset previous results
      setError('');
    }
  };

  // This function is called when the user clicks the "Run Attack" button
  const runAttack = async (e) => {
    e.preventDefault(); // Prevent the form from reloading the page
    if (!file) {
      setError('Please upload an image file first.');
      return;
    }

    setLoading(true);
    setResult(null);
    setError('');

    // FormData is the standard way to send files to a server
    const formData = new FormData();
    formData.append('image', file);
    formData.append('epsilon', epsilon);

    try {
      // This is the API call to your backend
      const response = await fetch('http://127.0.0.1:8000/attack', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data); // Store the results from the backend
    } catch (err) {
      console.error("Attack failed:", err);
      setError('Failed to run attack. Make sure the backend server is running.');
    } finally {
      setLoading(false); // Stop the loading indicator
    }
  };

  // --- JSX (The HTML structure of the page) ---
  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <h1 className={styles.title}>🔬 FGSM Adversarial Attack Demo</h1>
        <p className={styles.description}>
          Upload an image to see how a tiny, imperceptible perturbation can fool a ResNet18 model.
        </p>
        
        <form onSubmit={runAttack} className={styles.form}>
          <div className={styles.inputGroup}>
            <label htmlFor="imageUpload" className={styles.label}>1. Upload Image (PNG/JPEG)</label>
            <input 
              type="file" 
              id="imageUpload"
              accept="image/png, image/jpeg" 
              onChange={handleFileChange}
              className={styles.fileInput}
            />
          </div>

          <div className={styles.inputGroup}>
            <label htmlFor="epsilon" className={styles.label}>2. Set Epsilon (Perturbation): {epsilon.toFixed(3)}</label>
            <input 
              type="range"
              id="epsilon"
              min="0"
              max="0.2"
              step="0.005"
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              className={styles.slider}
            />
          </div>

          <button type="submit" disabled={loading || !file} className={styles.button}>
            {loading ? 'Attacking...' : '🚀 Run Attack'}
          </button>
        </form>
        
        {error && <p className={styles.error}>{error}</p>}
        
        {/* This section only appears after we get a result from the backend */}
        {result && (
          <div className={styles.results}>
            <div className={styles.statusBox} style={{ backgroundColor: result.attack_success ? '#28a745' : '#dc3545' }}>
              Attack Status: <strong>{result.attack_success ? 'Successful!' : 'Failed'}</strong>
            </div>

            <div className={styles.imageComparison}>
              <div className={styles.imageCard}>
                <h3>Original Image</h3>
                <img src={originalImage} alt="Original"/>
                <p>Model Prediction: <strong>{result.clean_prediction}</strong></p>
              </div>
              <div className={styles.imageCard}>
                <h3>Adversarial Image</h3>
                {/* The Base64 string from the API is used here to display the new image */}
                <img src={`data:image/png;base64,${result.adversarial_image_b64}`} alt="Adversarial"/>
                <p>Model Prediction: <strong>{result.adversarial_prediction}</strong></p>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}