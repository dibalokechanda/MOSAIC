// Intersection Observer for scroll animations
window.onload = () => {
    const observerOptions = { threshold: 0.1 };
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) entry.target.classList.add('visible');
        });
    }, observerOptions);

    document.querySelectorAll('.animate-up').forEach(el => observer.observe(el));
};

function copyCode(button, elementId) {
    const code = document.getElementById(elementId).innerText;
    navigator.clipboard.writeText(code).then(() => {
        const originalText = button.innerText;
        button.innerText = 'Copied!';
        button.classList.add('active');

        setTimeout(() => {
            button.innerText = originalText;
            button.classList.remove('active');
        }, 2000);
    });
}

/**
 * Handles switching between different stages of the forward pass walkthrough.
 * @param {number} stageIndex - The index of the stage to display (0-4).
 */
function switchStage(stageIndex) {
    console.log("Switching to stage:", stageIndex);

    // Update all sidebar cards
    const cards = document.querySelectorAll('.step-card');
    cards.forEach((card, index) => {
        if (index === stageIndex) {
            card.classList.add('active');
        } else {
            card.classList.remove('active');
        }
    });

    // Update all code viewport stages
    const stages = document.querySelectorAll('.code-stage');
    stages.forEach((stage, index) => {
        if (index === stageIndex) {
            stage.style.display = 'block';
            // Add a small delay for the fade-in animation to trigger
            stage.style.opacity = '0';
            setTimeout(() => {
                stage.style.opacity = '1';
                stage.style.transition = 'opacity 0.4s ease-in-out';
            }, 10);
        } else {
            stage.style.display = 'none';
        }
    });
}

/**
 * Interpretability Explorer Logic
 */
let interpretabilityData = {}; // Now loaded dynamically

let currentCohort = 'TCGA-NSCLC';
let currentSubclass = 'LUAD';

function setCohort(cohort) {
    console.log("[TextExplorer] Setting cohort:", cohort);
    currentCohort = cohort;
    currentSubclass = interpretabilityData[cohort][0];

    // Update Text Cohort Buttons specifically
    document.querySelectorAll('.interpret-section .cohort-btn').forEach(btn => {
        btn.classList.toggle('active', btn.innerText === cohort);
    });

    renderSubclassTabs();
    updateInterpretImage();
}

function renderSubclassTabs() {
    const container = document.getElementById('subclass-container');
    if (!container) return;
    container.innerHTML = '';

    interpretabilityData[currentCohort].forEach(sub => {
        const btn = document.createElement('button');
        btn.className = `subclass-btn ${sub === currentSubclass ? 'active' : ''}`;
        btn.innerText = sub;
        btn.onclick = () => setSubclass(sub);
        container.appendChild(btn);
    });
}

function setSubclass(subclass) {
    console.log("Setting subclass:", subclass);
    currentSubclass = subclass;

    // Update Subclass Buttons
    document.querySelectorAll('.subclass-btn').forEach(btn => {
        btn.classList.toggle('active', btn.innerText === subclass);
    });

    updateInterpretImage();
}

function updateInterpretImage() {
    const img = document.getElementById('interpret-img');
    if (!img) return;
    const newSrc = `text_interpretability/${currentCohort}/${currentSubclass}_text.png`;
    console.log("Loading image:", newSrc);

    // Fade out
    img.classList.remove('visible');

    setTimeout(() => {
        img.src = newSrc;
        img.onload = () => {
            img.classList.add('visible');
        };
        // Fallback if image fails to load or takes too long
        img.onerror = () => {
            console.error("Failed to load image:", newSrc);
            img.classList.add('visible'); // Show alt text at least
        };
    }, 150);
}

/**
 * Initialize Text Interpretability Explorer
 */
function initTextExplorer() {
    if (typeof textInterpretabilityManifest !== 'undefined') {
        interpretabilityData = textInterpretabilityManifest;
        
        const cohorts = Object.keys(interpretabilityData);
        if (cohorts.length > 0) {
            currentCohort = cohorts[0];
            currentSubclass = interpretabilityData[currentCohort][0];
            
            // Update Text Cohort Buttons specifically
            document.querySelectorAll('.interpret-section .cohort-btn').forEach(btn => {
                btn.classList.toggle('active', btn.innerText === currentCohort);
            });

            renderSubclassTabs();
            updateInterpretImage();
        }
    } else {
        console.error("[TextExplorer] Manifest variable 'textInterpretabilityManifest' not found.");
    }
}


/**
 * Image Interpretability Explorer Logic
 */
let visualHeatmapData = {}; // Now loaded dynamically

let currentVisualCohort = 'TCGA-NSCLC';
let currentVisualSlide = '';
let currentVisualVariation = 'combined';

function initVisualExplorer() {
    if (typeof visualHeatmapDataManifest !== 'undefined') {
        visualHeatmapData = visualHeatmapDataManifest;
        
        // Use manifest to set initial state
        const cohorts = Object.keys(visualHeatmapData);
        if (cohorts.length > 0) {
            currentVisualCohort = cohorts[0];
            currentVisualSlide = visualHeatmapData[currentVisualCohort][0];
            
            console.log("[VisualExplorer] Initializing with cohort:", currentVisualCohort, "and slide:", currentVisualSlide);
            
            populateSlideSelect();
            updateHeatmapImage();
        }
    } else {
        console.error("[VisualExplorer] Manifest variable 'visualHeatmapDataManifest' not found.");
    }
}

function setVisualCohort(cohort) {
    currentVisualCohort = cohort;
    currentVisualSlide = visualHeatmapData[cohort][0];

    // Update Cohort Buttons
    document.getElementById('v-cohort-nsclc').classList.toggle('active', cohort === 'TCGA-NSCLC');
    document.getElementById('v-cohort-rcc').classList.toggle('active', cohort === 'TCGA-RCC');

    populateSlideSelect();
    updateHeatmapImage();
}

function populateSlideSelect() {
    const select = document.getElementById('slide-select');
    if (!select) return;
    select.innerHTML = '';

    visualHeatmapData[currentVisualCohort].forEach(id => {
        const opt = document.createElement('option');
        opt.value = id;
        opt.innerText = id;
        select.appendChild(opt);
    });
    select.value = currentVisualSlide;
}

function setVisualSlide(slideId) {
    currentVisualSlide = slideId;
    updateHeatmapImage();
}

function setVisualVariation(variation) {
    currentVisualVariation = variation;

    // Update Cards
    document.querySelectorAll('.variation-card').forEach(card => {
        const titleText = card.querySelector('h3').innerText.toLowerCase().replace(' ', '').replace('-', '');
        const targetText = (variation === 'combined' ? 'chorus' : variation.toLowerCase()).replace('_', '');
        card.classList.toggle('active', titleText.includes(targetText));
    });

    updateHeatmapImage();
}

function updateHeatmapImage() {
    const img = document.getElementById('heatmap-img');
    if (!img) return;

    const newSrc = `Visual_Heatmaps/${currentVisualCohort}/${currentVisualSlide}/${currentVisualVariation}/uncertainty.png`;
    console.log("[VisualExplorer] Loading image:", newSrc);

    img.classList.remove('visible');

    // Set onload handler BEFORE setting src to catch cached images
    img.onload = () => {
        img.classList.add('visible');
    };
    
    img.onerror = () => {
        console.error("[VisualExplorer] Failed to load image at:", newSrc);
        img.src = 'https://via.placeholder.com/900x600?text=Heatmap+Not+Available';
        img.classList.add('visible');
    };

    setTimeout(() => {
        img.src = newSrc;
    }, 200);
}

// Global Initialization Update
document.addEventListener('DOMContentLoaded', () => {
    // Text Explorer Init
    if (document.getElementById('subclass-container')) {
        initTextExplorer();
    }

    // Visual Explorer Init
    if (document.getElementById('slide-select')) {
        initVisualExplorer();
    }

    // GCCA Algorithm Init
    if (document.getElementById('dots')) {
        initGCCA();
    }
});

/* GCCA Interactive Walkthrough Logic */
function initGCCA() {
    const gccaCards = [...document.querySelectorAll('.gcca-card')];
    const dotsEl = document.getElementById('dots');
    const stepLabel = document.getElementById('stepLabel');
    const prevBtn = document.getElementById('gccaPrevBtn');
    const nextBtn = document.getElementById('gccaNextBtn');
    
    if (!gccaCards.length || !dotsEl) return;

    let curGCCA = 0;
    const numSteps = gccaCards.length;

    // Build dots
    gccaCards.forEach((_, i) => {
        const d = document.createElement('div');
        d.className = 'progress-dot' + (i === 0 ? ' active' : '');
        d.onclick = () => { 
            if (i <= curGCCA + 1 || i < curGCCA) { 
                curGCCA = i; 
                renderGCCA(); 
            } 
        };
        dotsEl.appendChild(d);
    });
    
    const dots = [...dotsEl.children];

    function renderGCCA() {
        gccaCards.forEach((c, i) => {
            c.className = 'gcca-card ' + (i < curGCCA ? 'done' : i === curGCCA ? 'active' : 'locked');
        });
        
        dots.forEach((d, i) => {
            d.className = 'progress-dot' + (i < curGCCA ? ' done' : i === curGCCA ? ' active' : '');
        });

        stepLabel.textContent = `Step ${curGCCA + 1} / ${numSteps}`;
        prevBtn.disabled = curGCCA === 0;

        if (curGCCA === numSteps - 1) {
            nextBtn.innerHTML = 'Done <svg viewBox="0 0 24 24" style="width: 14px; height: 14px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;"><polyline points="20 6 9 17 4 12"/></svg>';
            nextBtn.disabled = true;
        } else {
            nextBtn.innerHTML = 'Next <svg viewBox="0 0 24 24" style="width: 14px; height: 14px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;"><polyline points="9 18 15 12 9 6"/></svg>';
            nextBtn.disabled = false;
        }

        // Auto-scroll to the active card
        gccaCards[curGCCA].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Re-render MathJax for the active card
        if (window.MathJax && MathJax.typesetPromise) {
            MathJax.typesetPromise([gccaCards[curGCCA]]).catch(() => {});
        }
    }

    window.goGCCA = (d) => {
        const n = curGCCA + d;
        if (n >= 0 && n < numSteps) { 
            curGCCA = n; 
            renderGCCA(); 
        }
    };

    renderGCCA();
}



// ============================
// CHORUS Variants Explorer
// ============================
const variantDB = {
    output: {
        title: 'Modulated Embedding  \\(\\tilde{\\boldsymbol{\\mu}}\\)',
        text: 'The uncertainty-modulated patch representation fed into the downstream FOCUS pipeline. Same shape as the fused mean: one vector per patch.',
        dims: '(N_s × 512) per slide',
        hl: 'a'
    },
    mu: {
        title: 'Fused Mean  \\(\\boldsymbol{\\mu}\\)',
        text: 'Cross-encoder average of the three projected representations. Each patch gets a single consensus vector combining what UNI, GigaPath, and ResNet-50 see.',
        dims: '(N_s × 512) image-dependent',
        hl: 'b'
    },
    rho: {
        title: 'Agreement Spectrum  \\(\\boldsymbol{\\rho}\\)',
        text: 'Measures how strongly all three encoders correlate on each latent dimension. High ρ_j means the dimension captures encoder-invariant tissue structure. Low ρ_j means encoder-specific noise.',
        dims: '(512,) image-independent, frozen',
        hl: 'c'
    },
    indicator: {
        title: 'Threshold Mask  \\(\\mathbb{1}[\\boldsymbol{\\rho} > \\tau]\\)',
        text: 'Binary mask keeping dimensions where agreement exceeds τ = 0.3 and zeroing out the rest. Invariant to small perturbations in ρ: a clean subspace selection.',
        dims: '(512,) binary {0, 1}',
        hl: 'd'
    },
    score: {
        title: 'Modified Selection Score',
        text: 'Final per-patch relevance score for FOCUS Stage 2 (token selection). Patches with high scores are kept; low-score patches are pruned from aggregation.',
        dims: 'scalar per patch',
        hl: 'a'
    },
    base_score: {
        title: 'Text–Patch Similarity  \\(\\mathbf{x}^\\top\\mathbf{T}\\)',
        text: 'Original FOCUS relevance: dot product between a patch feature and the text embedding matrix from class prompts. Measures how well a patch matches each cancer subtype description.',
        dims: 'x ∈ ℝ⁵¹², T ∈ ℝ⁵¹²ˣᶜ',
        hl: 'b'
    },
    beta: {
        title: 'Learnable Scale  \\(\\beta\\)',
        text: 'Learnable scalar (init 1.0) controlling how strongly the σ² penalty affects selection. The model learns end-to-end how much to trust the uncertainty signal.',
        dims: 'scalar: 1 of 2 new params in CHORUS',
        hl: 'c'
    },
    sigma_sum: {
        title: 'Total Patch Uncertainty  \\(\\sum_j \\sigma^2_j(x)\\)',
        text: 'Sum of per-dimension cross-encoder variance for patch x. Summing across dimensions makes this basis-invariant: preserved under any orthogonal rotation of the feature space.',
        dims: 'scalar per patch (from N_s × 512)',
        hl: 'd'
    },
    denom: {
        title: 'Precision Denominator  \\(1 + \\alpha\\sigma^2(x)\\)',
        text: 'When σ²(x) is low (encoders agree on this patch), the denominator stays near 1 and the global ρ weight passes through. When σ²(x) is high, it attenuates the feature: even if the dimension is globally reliable. Both signals must concur.',
        dims: '(N_s × 512): α learnable, init 1.0',
        hl: 'd'
    },
    inv_rho: {
        title: 'Inverted Agreement  \\((1 - \\boldsymbol{\\rho})\\)',
        text: 'Flips the signal: high-agreement dimensions are downweighted, low-agreement dimensions amplified.',
        dims: '(512,) same shape as ρ, inverted',
        hl: 'd'
    },
    attn_out: {
        title: 'Attention Output',
        text: 'Result of the ρ-modulated cross-modal attention in FOCUS Stage 4. Text-prompt queries attend to patch features, weighted by the agreement-adjusted similarity.',
        dims: '(n_prompts × 512)',
        hl: 'a'
    },
    qk: {
        title: '\\(\\rho\\)-weighted Q·K Scores',
        text: 'Attention queries Q and keys K are each scaled by √ρ before the dot product. This is equivalent to computing attention in a ρ-weighted inner-product space: high-agreement dimensions contribute more to the similarity score.',
        dims: 'Q, K ∈ ℝⁿˣ⁶⁴ per head',
        hl: 'b'
    },
    sqrt_dh: {
        title: 'Head Dimension Scale  \\(\\sqrt{d_h}\\)',
        text: 'Standard scaled dot-product attention normalization (d_h = 64). Prevents attention logits from growing too large as the head dimension increases.',
        dims: 'd_h = 64 (8 heads × 64 = 512)',
        hl: 'c'
    },
    values: {
        title: 'Value Vectors  \\(\\mathbf{V}\\)',
        text: 'Unmodified value projections of the patch features. In this variant, ρ modulates only which patches get attended to (Q/K), not the information that flows through (V).',
        dims: 'V ∈ ℝⁿˣ⁶⁴ per head',
        hl: 'e'
    },
    values_mod: {
        title: 'Modulated Values  \\(\\mathbf{V} \\odot \\sqrt{\\boldsymbol{\\rho}}\\)',
        text: 'Both the attention weighting (Q/K) and the content (V) are modulated by the global agreement signal, ensuring only consistent features flow to the final representation.',
        dims: 'V ∈ ℝⁿˣ⁶⁴ — agreement scaled',
        hl: 'e'
    }
};

const vHlColors = {
    a: '#818cf8', b: '#34d399', c: '#fb923c', d: '#f472b6', e: '#2dd4bf'
};

let activeVTip = null;

function toggleVCard(id) {
    document.getElementById(id).classList.toggle('open');
}

function showVTip(el) {
    const cardId = el.getAttribute('data-card');
    const key = el.getAttribute('data-key');
    const info = variantDB[key];
    if (!info) return;

    const card = document.getElementById(cardId);
    card.querySelectorAll('.v-eq-part').forEach(p => p.classList.remove('selected'));
    el.classList.add('selected');

    const tip = document.getElementById(cardId + '-tip');
    const dotColor = vHlColors[info.hl] || '#999';

    tip.innerHTML = `
        <button class="v-tip-close" onclick="hideVTip('${cardId}')">&times;</button>
        <div class="vt-title">
            <span class="vt-dot" style="background:${dotColor}"></span>
            ${info.title}
        </div>
        <div class="vt-text">${info.text}</div>
        <div class="vt-dims">${info.dims}</div>
    `;

    tip.classList.add('visible');
    activeVTip = cardId;

    if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise([tip]).catch(() => {});
    }
}

function hideVTip(cardId) {
    const tip = document.getElementById(cardId + '-tip');
    tip.classList.remove('visible');
    const card = document.getElementById(cardId);
    card.querySelectorAll('.v-eq-part').forEach(p => p.classList.remove('selected'));
    activeVTip = null;
}

function clickVLabel(el, cardId, key) {
    const card = document.getElementById(cardId);
    if (!card.classList.contains('open')) card.classList.add('open');
    const part = card.querySelector(`.v-eq-part[data-key="${key}"]`);
    if (part) showVTip(part);
}

document.addEventListener('click', (e) => {
    if (!e.target.closest('.v-eq-part') && !e.target.closest('.v-tip') && !e.target.closest('.v-eq-arrow-label')) {
        if (activeVTip) hideVTip(activeVTip);
    }
});
