// Event listener for the "Multilingual Text Summarizer" button click
document.getElementById("heading-button").addEventListener("click", function() {
    // Reload the page when the button is clicked
    location.reload();
});

// Add loading state management functions
function setLoadingState(isLoading) {
    const button = document.getElementById("generate-summary");
    const spinner = document.getElementById("spinner");
    const form = document.querySelector(".form-section");
    const outputs = document.querySelectorAll(".output-box");
    
    if (isLoading) {
        button.innerHTML = "Generating<span class='loading-text'>...</span>";
        button.disabled = true;
        spinner.style.display = "block";
        form.classList.add("loading");
        outputs.forEach(output => output.classList.add("loading"));
    } else {
        button.innerHTML = "✨ Generate Summary";
        button.disabled = false;
        spinner.style.display = "none";
        form.classList.remove("loading");
        outputs.forEach(output => output.classList.remove("loading"));
    }
}

// Add copy functionality
function addCopyFunctionality() {
    const copyButton = document.getElementById('copy-button');
    const summaryOutput = document.getElementById('summary-output');

    copyButton.addEventListener('click', async () => {
        try {
            const text = summaryOutput.innerText;
            await navigator.clipboard.writeText(text);
            
            // Visual feedback
            copyButton.textContent = '✅ Copied!';
            copyButton.classList.add('copy-success');
            
            // Reset button text after 2 seconds
            setTimeout(() => {
                copyButton.textContent = '📋 Copy';
                copyButton.classList.remove('copy-success');
            }, 2000);
        } catch (err) {
            console.error('Failed to copy text:', err);
            copyButton.textContent = '❌ Failed';
            
            setTimeout(() => {
                copyButton.textContent = '📋 Copy';
            }, 2000);
        }
    });
}

// Call this function after the DOM is loaded
document.addEventListener('DOMContentLoaded', addCopyFunctionality);

// Update the summary display function to ensure the copy button is visible only when there's content
function updateSummaryDisplay(summary) {
    const summaryOutput = document.getElementById('summary-output');
    const copyButton = document.getElementById('copy-button');
    
    summaryOutput.innerText = summary;
    copyButton.style.display = summary && summary !== 'Generating summary...' ? 'block' : 'none';
}

// Event listener for the "Generate Summary" button click
document.getElementById("generate-summary").addEventListener("click", async () => {
    const language = document.getElementById("language").value;
    const article = document.getElementById("article").value.trim();
    const referenceSummary = document.getElementById("reference-summary").value.trim();
    const summaryType = document.getElementById("summary-type").value;

    // Check if article text is provided
    if (!article) {
        alert("🚨 Please enter some text to summarize!");
        return;
    }

    // Prepare the data to send to the backend
    const requestData = {
        language: language,
        article: article,
        reference_summary: referenceSummary,
        summary_type: summaryType,
    };

    try {
        // Set loading state
        setLoadingState(true);
        
        // Reset output displays
        document.getElementById("summary-output").innerText = "Generating summary...";
        document.getElementById("evaluation-output").innerText = "Calculating scores...";

        // Call the backend API to generate the summary
        const response = await fetch("/summarize", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestData),
        });

        const data = await response.json();

        // Check for errors in the response
        if (data.error) {
            throw new Error(data.error);
        }

        // Update summary display
        updateSummaryDisplay(data.summary || "⚠️ Error generating summary.");

        // Display BERTScore evaluation if available
        if (data.bert_score) {
            document.getElementById("evaluation-output").innerHTML = `
                <p><strong>Precision:</strong> ${data.bert_score.P}</p>
                <p><strong>Recall:</strong> ${data.bert_score.R}</p>
                <p><strong>F1 Score:</strong> ${data.bert_score.F1}</p>
            `;
        } else {
            document.getElementById("evaluation-output").innerText = "⚠️ No BERTScore evaluation available.";
        }
    } catch (error) {
        console.error("❌ Error connecting to the backend:", error);
        updateSummaryDisplay("❌ Error: " + error.message);
        document.getElementById("evaluation-output").innerText = "⚠️ Evaluation unavailable due to error.";
    } finally {
        // Reset loading state
        setLoadingState(false);
    }
});

// Add this debugging function
function testUIComponents() {
    console.log("Testing UI Components...");
    
    // Test form elements
    const elements = {
        languageSelect: document.getElementById("language"),
        summaryTypeSelect: document.getElementById("summary-type"),
        articleTextarea: document.getElementById("article"),
        referenceSummaryTextarea: document.getElementById("reference-summary"),
        generateButton: document.getElementById("generate-summary"),
        summaryOutput: document.getElementById("summary-output"),
        evaluationOutput: document.getElementById("evaluation-output"),
        copyButton: document.getElementById("copy-button"),
        spinner: document.getElementById("spinner")
    };

    // Log status of each element
    for (const [name, element] of Object.entries(elements)) {
        console.log(`${name}: ${element ? '✅ Found' : '❌ Missing'}`);
    }

    // Test loading state
    console.log("\nTesting loading state...");
    setLoadingState(true);
    setTimeout(() => {
        setLoadingState(false);
        console.log("Loading state test complete");
    }, 2000);

    // Test copy functionality
    console.log("\nTesting copy button visibility...");
    updateSummaryDisplay("Test summary");
    console.log("Copy button should be visible now");
}

// Add these functions after your existing code
function setupInputToggle() {
    const textInputBtn = document.getElementById('text-input-btn');
    const fileInputBtn = document.getElementById('file-input-btn');
    const textSection = document.getElementById('text-input-section');
    const fileSection = document.getElementById('file-input-section');

    textInputBtn.addEventListener('click', () => {
        textInputBtn.classList.add('active');
        fileInputBtn.classList.remove('active');
        textSection.style.display = 'block';
        fileSection.style.display = 'none';
    });

    fileInputBtn.addEventListener('click', () => {
        fileInputBtn.classList.add('active');
        textInputBtn.classList.remove('active');
        fileSection.style.display = 'block';
        textSection.style.display = 'none';
    });
}

async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('File upload failed');
        }
        
        const data = await response.json();
        document.getElementById('article').value = data.text;
        
        // Switch back to text input view
        document.getElementById('text-input-btn').click();
        
    } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error uploading file: ' + error.message);
    }
}

// Update the DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', () => {
    addCopyFunctionality();
    setupInputToggle();
    
    // Add file upload handler
    document.getElementById('file-upload').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });
    
    // Add debug shortcut
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.shiftKey && e.key === 'T') {
            testUIComponents();
        }
    });

    // Update the file input accept attribute
    document.getElementById('file-upload').accept = '.txt,.docx,.pdf';
});