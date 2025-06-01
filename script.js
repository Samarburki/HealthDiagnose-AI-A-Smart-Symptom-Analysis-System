// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;
        
        const targetElement = document.querySelector(targetId);
        if (targetElement) {
            const navbar = document.querySelector('nav.navbar');
            const offset = navbar ? navbar.offsetHeight : 0;
            
            window.scrollTo({
                top: targetElement.offsetTop - offset,
                behavior: 'smooth'
            });
            
            history.pushState(null, null, targetId);
        }
    });
});

// Form submission handler for diagnosis form
document.getElementById('diagnosisForm')?.addEventListener('submit', function() {
    // Set flag to scroll after reload
    sessionStorage.setItem('shouldScrollToResults', 'true');
    
    // Immediately show results section
    const results = document.querySelector('.results-section');
    if (results) {
        results.style.display = 'block';
    }
});

// Form submission handler for evaluation survey form
document.getElementById('surveyForm')?.addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent default form submission

    const rating = document.getElementById('rating').value;
    const feedback = document.getElementById('feedback').value;

    if (rating) {
        console.log('Survey Submitted!');
        console.log('Rating:', rating);
        console.log('Feedback:', feedback);

        // Here you would typically send this data to a server
        // using fetch() or XMLHttpRequest.
        // Example:
        /*
        fetch('/api/submit-feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ rating, feedback }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            alert('Thank you for your feedback!');
            document.getElementById('surveyForm').reset(); // Clear the form
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('There was an error submitting your feedback.');
        });
        */

        alert('Thank you for your feedback!');
        document.getElementById('surveyForm').reset(); // Clear the form after submission
    } else {
        alert('Please select a rating before submitting.');
    }
});


// Optimized scroll to results function
function scrollToResults() {
    if (sessionStorage.getItem('shouldScrollToResults') === 'true') {
        sessionStorage.removeItem('shouldScrollToResults');
        
        const results = document.querySelector('.results-section');
        if (results) {
            // 👇 Ensure results are visible before scrolling
            results.style.display = 'block';

            const navbar = document.querySelector('nav.navbar');
            const offset = navbar ? navbar.offsetHeight + 20 : 20;
            const currentPos = window.scrollY;
            const targetPos = results.offsetTop - offset;

            // Scroll smoothly only if needed
            if (targetPos > currentPos) {
                window.scrollTo({
                    top: targetPos,
                    behavior: 'smooth'
                });
            }
        }
    }
}


// Initialize when page loads
window.addEventListener('load', function() {
    // Check if we need to scroll to results
    setTimeout(scrollToResults, 100); // Small delay
    
    // Other initializations...
    setupNavHighlighting();
    setupInputEffects();
});

// Also check when DOM is loaded
document.addEventListener('DOMContentLoaded', scrollToResults);