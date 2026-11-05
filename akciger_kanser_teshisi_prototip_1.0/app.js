document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // 15 Form verisinin tamamını topla
    const patientData = {
        gender: parseInt(document.getElementById('gender').value),
        age: parseInt(document.getElementById('age').value),
        smoking: parseInt(document.getElementById('smoking').value),
        yellow_fingers: parseInt(document.getElementById('yellow_fingers').value),
        anxiety: parseInt(document.getElementById('anxiety').value),
        peer_pressure: parseInt(document.getElementById('peer_pressure').value),
        chronic_disease: parseInt(document.getElementById('chronic_disease').value),
        fatigue: parseInt(document.getElementById('fatigue').value),
        allergy: parseInt(document.getElementById('allergy').value),
        wheezing: parseInt(document.getElementById('wheezing').value),
        alcohol: parseInt(document.getElementById('alcohol').value),
        coughing: parseInt(document.getElementById('coughing').value),
        shortness_of_breath: parseInt(document.getElementById('shortness_of_breath').value),
        swallowing: parseInt(document.getElementById('swallowing').value),
        chest_pain: parseInt(document.getElementById('chest_pain').value)
    };

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patientData)
        });
        
        const result = await response.json();
        
        const resultBox = document.getElementById('result-box');
        const riskStatus = document.getElementById('risk-status');
        const riskProb = document.getElementById('risk-prob');
        
        riskStatus.innerText = result.status;
        riskStatus.style.color = result.status === "YÜKSEK RİSK" ? "#ff4757" : "#2ed573";
        riskProb.innerText = `Olasılık: ${result.probability}`;
        
        resultBox.classList.remove('hidden');
    } catch (error) {
        alert("API'ye bağlanılamadı. FastAPI sunucusunun çalıştığından emin olun.");
    }
});