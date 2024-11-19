//popup.js

// Add event listener for the "Analyze" button
document.getElementById('checkBtn').addEventListener('click', function() {
  const inputText = document.getElementById('inputText').value;
  const checkBtn = document.getElementById('checkBtn');
  const resultElement = document.getElementById('result');

  // Check if input is empty
  if (!inputText.trim()) {
    resultElement.textContent = "Please enter text to analyze.";
    resultElement.className = ""; // Clear any previous styling
    return;
  }

  // Disable the button and show "Analyzing" message
  checkBtn.disabled = true;
  resultElement.textContent = "Analyzing...";
  resultElement.className = ""; // Clear any previous styling

  // Fetch request to the server
  fetch("http://localhost:5001/check", {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: inputText }),
  })
    .then(response => response.json())
    .then(data => {
      // Display result based on server response
      resultElement.textContent = data.is_fake ? "This is likely not credible" : "This is likely credible";
      resultElement.className = data.is_fake ? "non-credible" : "credible"; // Add appropriate styling
    })
    .catch(error => {
      console.error('Error:', error);
      resultElement.textContent = 'Error analyzing text. Please try again.';
    })
    .finally(() => {
      // Re-enable the button after analysis is complete
      checkBtn.disabled = false;
    });
});
