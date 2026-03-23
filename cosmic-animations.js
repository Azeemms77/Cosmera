// cosmic-animations.js

document.addEventListener('DOMContentLoaded', () => {
    // 1 & 4. Inject animated UI container
    const cosmicUI = document.createElement('div');
    cosmicUI.id = 'cosmic-animations-container';
    cosmicUI.style.position = 'fixed';
    cosmicUI.style.top = '0';
    cosmicUI.style.left = '0';
    cosmicUI.style.width = '100%';
    cosmicUI.style.height = '100%';
    cosmicUI.style.pointerEvents = 'none';
    cosmicUI.style.zIndex = '-1';
    
    // 2. Floating Rocket
    const rocket = document.createElement('div');
    rocket.className = 'floating-rocket';
    rocket.innerHTML = '🚀';
    cosmicUI.appendChild(rocket);
    
    // 3. Orbiting Planet
    const orbitContainer = document.createElement('div');
    orbitContainer.className = 'orbiting-planet-container';
    orbitContainer.innerHTML = `
        <div class="planet">🪐</div>
        <div class="moon-orbit">
            <div class="moon">🌕</div>
        </div>
    `;
    cosmicUI.appendChild(orbitContainer);

    // 4. Moving Comets
    for (let i = 0; i < 3; i++) {
        const comet = document.createElement('div');
        comet.className = 'comet';
        cosmicUI.appendChild(comet);
    }
    
    // 1. Satellites / other elements
    const satellite = document.createElement('div');
    satellite.className = 'space-object';
    satellite.innerHTML = '🛰️';
    satellite.style.top = '70%';
    satellite.style.left = '10%';
    satellite.style.fontSize = '35px';
    satellite.style.animation = 'floatAstronaut 6s ease-in-out infinite';
    cosmicUI.appendChild(satellite);

    const alien = document.createElement('div');
    alien.className = 'space-object';
    alien.innerHTML = '🛸';
    alien.style.top = '15%';
    alien.style.left = '80%';
    alien.style.fontSize = '40px';
    alien.style.animation = 'floatAstronaut 8s ease-in-out infinite reverse';
    cosmicUI.appendChild(alien);

    document.body.appendChild(cosmicUI);

    // 5. Interactive Hover Animation
    const hoverElements = document.querySelectorAll('.gallery-card, .fact-card, .compare-box, .knowledge-card, .glossary-item');
    hoverElements.forEach(el => {
        const star1 = document.createElement('i');
        star1.className = 'fas fa-star hover-star hover-star-1';
        const star2 = document.createElement('i');
        star2.className = 'fas fa-star hover-star hover-star-2';
        el.appendChild(star1);
        el.appendChild(star2);
    });

    // 6. Animated Astronaut for Chat Panel
    const chatPanel = document.getElementById('chatPanel');
    if (chatPanel) {
        // Place astronaut container inside chatPanel relative
        chatPanel.style.position = 'fixed'; // It already is
        const astronaut = document.createElement('div');
        astronaut.className = 'chat-astronaut';
        astronaut.innerHTML = '👨‍🚀';
        chatPanel.appendChild(astronaut);
    }

    // 7. Parallax Space Background
    const parallaxLayer = document.createElement('div');
    parallaxLayer.id = 'parallax-starfield';
    parallaxLayer.innerHTML = `
        <div class="layer layer-1" id="layer1"></div>
        <div class="layer layer-2" id="layer2"></div>
        <div class="layer layer-3" id="layer3"></div>
    `;
    document.body.appendChild(parallaxLayer);

    const generateStars = (layerId, count, sizeMultiplier) => {
        const layer = document.getElementById(layerId);
        if (!layer) return;
        for (let i = 0; i < count; i++) {
            const star = document.createElement('div');
            star.style.position = 'absolute';
            star.style.background = 'white';
            star.style.borderRadius = '50%';
            const size = Math.random() * sizeMultiplier;
            star.style.width = size + 'px';
            star.style.height = size + 'px';
            star.style.left = Math.random() * 100 + '%';
            star.style.top = Math.random() * 100 + '%';
            star.style.opacity = Math.random() * 0.8 + 0.2;
            layer.appendChild(star);
        }
    };

    generateStars('layer1', 30, 2);
    generateStars('layer2', 20, 3);
    generateStars('layer3', 15, 4);

    window.addEventListener('scroll', () => {
        const scrollY = window.scrollY;
        const l1 = document.getElementById('layer1');
        const l2 = document.getElementById('layer2');
        const l3 = document.getElementById('layer3');
        if(l1) l1.style.transform = `translateY(${scrollY * -0.1}px)`;
        if(l2) l2.style.transform = `translateY(${scrollY * -0.2}px)`;
        if(l3) l3.style.transform = `translateY(${scrollY * -0.3}px)`;
    });
});
