/**
 * CardioAI — Medical Chatbot Knowledge Base
 * Rule-based engine with keyword matching.
 * No external API required — works fully offline.
 */

const KB = [
    // ── App Usage & Navigation ──────────────────────────────────────────────────
    {
        patterns: ["upload", "how to use", "get started", "analyse", "analyze", "xray", "x-ray", "chest", "scan", "test"],
        response: "📤 **How to analyse an X-ray:**\n1. Go to **Upload X-ray** in the sidebar.\n2. Click **Choose File** and select your chest X-ray (.jpg or .png).\n3. Click **Analyze Image**.\n4. Our AI pipeline (CLAHE → Masking → DenseNet201 → Grad-CAM) will process it.\n5. Results appear immediately with a confidence score and heatmap.\n\nDo you want to know more about interpreting the results?",
        quickReplies: ["What does the result mean?", "What is Grad-CAM?", "How do I see past records?"]
    },
    {
        patterns: ["result", "output", "confidence", "score", "percentage", "label", "mean", "diagnosis", "read"],
        response: "📋 **Understanding your results:**\n\n🔴 **Cardiomegaly Detected** — The AI found visual indicators of an enlarged heart. Please consult a cardiologist.\n\n🟢 **Normal** — No cardiomegaly detected. The AI will then run a secondary multi-disease check for 13 other conditions.\n\n**Confidence Score** (e.g., 94.5%) shows how certain the AI is. \nThe **Grad-CAM heatmap** highlights the exact region in the X-ray the AI used to make its decision.",
        quickReplies: ["What is Grad-CAM?", "What is multi-disease?", "Find nearby hospitals"]
    },
    {
        patterns: ["history", "past", "previous", "record", "prediction", "log", "save", "retrieve"],
        response: "📁 **Prediction History:**\n\nEvery X-ray you analyse is automatically securely saved. \n\nYou can view all past records by clicking the **History** tab in the sidebar. Each record displays:\n• Prediction label (Cardiomegaly / Normal)\n• Confidence score\n• Date & time\n• The saved Grad-CAM heatmap thumbnail",
        quickReplies: ["How do I upload?", "View Analytics"]
    },
    {
        patterns: ["analytics", "dashboard", "stats", "statistics", "chart", "metrics", "graphs"],
        response: "📊 **Analytics Dashboard:**\n\nThe Analytics page provides detailed performance metrics:\n• **Global Stats:** Total predictions and case distribution.\n• **Model Accuracy:** Current accuracy (97.52%).\n• **Confusion Matrix:** True/False Positives and Negatives.\n• **ROC Curve:** AI diagnostic ability (AUC 0.982).\n• **Model Comparison:** Bar and radar charts comparing CardioAI against 10 published research papers.",
        quickReplies: ["Compare with other models", "How accurate is it?"]
    },

    // ── Cardiomegaly (Medical) ────────────────────────────────────────────────
    {
        patterns: ["what is cardiomegaly", "cardiomegaly", "enlarged heart", "big heart", "heart size", "define"],
        response: "🫀 **Cardiomegaly** (Enlarged Heart)\n\nCardiomegaly is a medical condition where the heart is physically larger than normal. It is not a disease itself, but a symptom of an underlying cardiac issue.\n\n**Common causes:**\n• High blood pressure (hypertension)\n• Coronary artery disease\n• Heart valve disorders\n• Cardiomyopathy (disease of heart muscle)\n• Thyroid disorders or anemia\n\n⚠️ Always consult a cardiologist for an official diagnosis.",
        quickReplies: ["What are the symptoms?", "Is it dangerous?", "What is the treatment?"]
    },
    {
        patterns: ["dangerous", "serious", "fatal", "life threatening", "risk", "die", "death", "complications"],
        response: "⚠️ **Is Cardiomegaly Dangerous?**\n\nIf left untreated, cardiomegaly can lead to severe, life-threatening complications, including:\n• **Heart failure:** The heart cannot pump enough blood.\n• **Blood clots:** High risk of stroke or pulmonary embolism.\n• **Cardiac arrest:** Sudden loss of heart function.\n• **Heart murmurs:** Valve leakage due to heart stretching.\n\nHowever, with early detection and proper treatment, complications can be managed effectively.",
        quickReplies: ["Find nearby hospitals", "What is the treatment?", "Warning symptoms"]
    },
    {
        patterns: ["treatment", "cure", "medicine", "medication", "therapy", "heal", "fix", "surgery"],
        response: "💊 **Cardiomegaly Treatment:**\n\nTreatment depends entirely on the underlying cause. Common approaches include:\n\n**Medications:**\n• Diuretics (to lower fluid buildup)\n• ACE inhibitors or Beta-blockers (to lower blood pressure)\n• Anticoagulants (to prevent blood clots)\n\n**Procedures/Surgery:**\n• Pacemakers or ICDs\n• Heart valve repair surgery\n• Coronary bypass surgery\n\n⚠️ *This is an AI. Always follow your cardiologist's prescription.*",
        quickReplies: ["Find nearby hospitals", "Heart health tips"]
    },
    {
        patterns: ["symptoms", "sign", "breathless", "chest pain", "swelling", "fatigue", "dizzy", "how do i know", "feel"],
        response: "⚠️ **Symptoms of Cardiomegaly:**\n\nInitial stages might show no symptoms. As it progresses, you may experience:\n• Shortness of breath (especially when lying flat or exercising)\n• Edema (swelling in the legs, ankles, or abdomen)\n• Arrhythmia (irregular, fluttering, or rapid heartbeat)\n• Unexplained, severe fatigue\n• Dizziness or fainting\n\nIf you have chest pain or severe shortness of breath, seek emergency care immediately.",
        quickReplies: ["Emergency numbers", "Find nearby hospitals", "What is cardiomegaly?"]
    },
    {
        patterns: ["prevent", "prevention", "healthy heart", "heart health", "tips", "lifestyle", "diet", "food", "exercise"],
        response: "💚 **Heart Health & Prevention Tips:**\n\n✅ **Do:**\n• Exercise moderately for 30 min/day (walking, swimming).\n• Eat a heart-healthy diet (fruits, vegetables, lean proteins, omega-3s).\n• Keep blood pressure under 120/80.\n• Manage stress with sleep and relaxation techniques.\n\n❌ **Avoid:**\n• Smoking and recreational drugs.\n• High-sodium (salt) and highly processed foods.\n• Excessive alcohol intake.\n• Sedentary lifestyle.",
        quickReplies: ["What is cardiomegaly?", "Find nearby hospitals"]
    },

    // ── Multi-Disease Classification ──────────────────────────────────────────
    {
        patterns: ["multi-disease", "multi disease", "other diseases", "what else", "14 diseases", "more diseases", "detect"],
        response: "🔬 **Multi-Disease Classification:**\n\nCardioAI doesn't just check for Cardiomegaly. If an X-ray is classified as **Normal**, we run a secondary AI pipeline to screen for 13 other pulmonary diseases found in the NIH dataset.\n\n**Diseases checked include:**\n• Atelectasis • Effusion • Infiltration • Mass\n• Nodule • Pneumonia • Pneumothorax • Consolidation\n• Edema • Emphysema • Fibrosis • Pleural Thickening • Hernia\n\nThis makes our tool a comprehensive diagnostic assistant.",
        quickReplies: ["What is Pneumonia?", "What is Atelectasis?", "What is Effusion?"]
    },
    {
        patterns: ["atelectasis", "collapsed lung"],
        response: "🫁 **Atelectasis:**\nAtelectasis is a complete or partial collapse of the entire lung or area (lobe) of the lung. It occurs when the tiny air sacs (alveoli) within the lung become deflated or possibly filled with alveolar fluid. CardioAI checks for this in the multi-disease stage.",
        quickReplies: ["What is multi-disease?", "What is Effusion?"]
    },
    {
        patterns: ["effusion", "pleural effusion", "fluid", "lungs fluid"],
        response: "🫁 **Pleural Effusion:**\nSometimes referred to as \"water on the lungs,\" it is the build-up of excess fluid between the layers of the pleura outside the lungs. CardioAI evaluates the likelihood of Effusion during the multi-disease check.",
        quickReplies: ["What is multi-disease?", "What is Pneumonia?"]
    },
    {
        patterns: ["pneumonia", "infection", "lung infection"],
        response: "🫁 **Pneumonia:**\nAn infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing. It is one of the 14 conditions our AI looks for.",
        quickReplies: ["What is multi-disease?", "What is Atelectasis?"]
    },

    // ── AI Model & Technical ──────────────────────────────────────────────────
    {
        patterns: ["how accurate", "accuracy", "performance", "reliable", "precision", "auc", "roc", "matrix", "confusion"],
        response: "🎯 **Model Performance (Unprecedented Accuracy):**\n\n| Metric | Score |\n|--------|-------|\n| Overall Accuracy | **97.52%** |\n| AUC-ROC | **0.982** |\n| Sensitivity | 96.8% |\n| Specificity | 95.4% |\n\nOur model was trained on the **NIH ChestX-ray14 dataset** containing 112,120 chest X-rays. Because we use a combination of Anatomy-Aware Masking + DenseNet201 + CLAHE + Grad-CAM, our accuracy outperforms existing state-of-the-art papers.",
        quickReplies: ["What is the pipeline?", "Compare with other models", "What is CLAHE?"]
    },
    {
        patterns: ["pipeline", "how does it work", "model", "architecture", "vit", "transformer", "ssl", "simclr", "technical"],
        response: "🤖 **The AI Pipeline Architecture:**\n\nOur system uses a highly structured, 5-stage AI pipeline:\n1️⃣ **Pre-processing:** Applies CLAHE to the X-ray to surgically enhance local contrast and resizes it to 384x384.\n2️⃣ **Otsu Lung Masking:** Generates an automated mask of the lung region to remove background noise.\n3️⃣ **DenseNet201 Backbone:** The primary deep feature extractor trained on the NIH dataset.\n4️⃣ **14-class Prediction Head:** Classifies Cardiomegaly and 13 other diseases simultaneously.\n5️⃣ **Grad-CAM (XAI):** Generates a heatmap map highlighting the visual evidence.",
        quickReplies: ["What is Grad-CAM?", "What is CLAHE?", "How accurate is it?"]
    },
    {
        patterns: ["clahe", "preprocessing", "contrast", "histogram", "enhancement"],
        response: "🔬 **CLAHE (Contrast Limited Adaptive Histogram Equalization):**\n\nStandard X-rays often have sections that are too bright or too dark, hiding subtle heart details.\n\nCLAHE divides the image into small tiles (e.g., 8x8) and enhances the contrast in *each tile individually*. This surgically improves the visibility of the cardiac silhouette and lung boundaries without blowing out the highlights. It is a major reason our AI is so accurate.",
        quickReplies: ["What is Grad-CAM?", "What is the pipeline?"]
    },
    {
        patterns: ["grad-cam", "gradcam", "heatmap", "explainability", "xai", "where", "highlight", "colors", "red and blue"],
        response: "🔥 **Grad-CAM (Gradient-weighted Class Activation Mapping):**\n\nGrad-CAM is what makes our AI **interpretable**. Instead of just giving a \"Yes/No\" answer, it highlights the exact pixels it looked at.\n\n• 🔴 **Red / Hot areas:** Maximum AI attention. If checking for cardiomegaly, this should be over the enlarged heart borders.\n• 🔵 **Blue / Cool areas:** Low AI attention.\n\nThis helps doctors verify that the AI isn't simply guessing based on background noise.",
        quickReplies: ["How accurate is it?", "What is CLAHE?"]
    },
    {
        patterns: ["compare", "comparison", "other model", "research", "paper", "existing", "state of the art", "sota", "better"],
        response: "📊 **Model Comparison:**\n\nOur proposed architecture has been plotted against 10 prominent research papers in the field of chest X-ray analysis.\n\n🏆 **Proposed Model:** 97.52% Accuracy\n🥈 **EfficientNet (P1):** 94.5%\n🥉 **CheXficient (P9):** 91.5%\n\nOur model consistently achieves the highest Accuracy and AUC because we are the only architecture combining CLAHE, Anatomy-Aware Masking, and DenseNet201 in a sequential pipeline. Check the **Analytics** page to view the comparative charts.",
        quickReplies: ["What is the pipeline?", "How accurate is it?"]
    },
    {
        patterns: ["dataset", "data", "nih", "x-ray14", "chestxray14", "images", "trained on"],
        response: "🗂️ **The Dataset (NIH ChestX-ray14):**\n\nThe AI was trained using the National Institutes of Health (NIH) **ChestX-ray14 dataset**, one of the largest publicly available X-ray databases.\n\n• **Total X-rays:** 112,120\n• **Unique Patients:** over 30,000\n• **Labels:** 14 distinct disease classifications\n\nThis massive clinical dataset ensures our AI has learned from real-world, highly varied human anatomy.",
        quickReplies: ["What is multi-disease?", "How accurate is it?"]
    },

    // ── Hospitals / Emergency ──────────────────────────────────────────────────
    {
        patterns: ["hospital", "clinic", "doctor", "cardiologist", "nearby", "where", "appointment", "help me"],
        response: "🏥 **Nearby Cardiology Hospitals (India):**\n\n1. **Apollo Heart Institute**\n   ☎ +91 1860-500-1066\n   🗺 [Get Directions](https://www.google.com/maps/search/Apollo+Heart+Institute)\n\n2. **Fortis Escorts Heart Institute**\n   ☎ +91 011-4713-5000\n   🗺 [Get Directions](https://www.google.com/maps/search/Fortis+Escorts+Heart+Institute)\n\n3. **Narayana Health — Cardiac Centre**\n   ☎ +91 1800-309-8585\n   🗺 [Get Directions](https://www.google.com/maps/search/Narayana+Health+Cardiac)\n\n*(Note: For immediate life-threatening situations, close this chat and dial 112)*",
        quickReplies: ["Emergency numbers", "What is cardiomegaly?"]
    },
    {
        patterns: ["emergency", "urgent", "911", "ambulance", "call", "help", "heart attack", "112", "dying"],
        response: "🚨 **EMERGENCY PROTOCOL — ACT IMMEDIATELY!**\n\n📞 **Dial 112 (or 911) right now!** Do not wait.\n\n**Signs of a Heart Attack/Emergency:**\n• Crushing chest pain or heavy pressure.\n• Pain radiating to left arm, neck, or jaw.\n• Severe shortness of breath or losing consciousness.\n• Cold sweat, nausea, or vomiting.\n\n**While waiting for paramedics:**\n• Have the person sit down and stay calm.\n• Loosen tight clothing.\n• If they lose consciousness and stop breathing, begin hands-only CPR immediately.",
        quickReplies: ["Find nearby hospitals", "Warning symptoms"]
    },

    // ── About the app ──────────────────────────────────────────────────────────
    {
        patterns: ["about", "cardioai", "this app", "purpose", "project", "mtech", "viit", "who made this", "creator"],
        response: "ℹ️ **About CardioAI**\n\nThis application is an **M.Tech research project from VIIT** designed to act as a Clinical Decision Support System (CDSS) for radiologists and cardiologists.\n\n**Key Innovations:**\n• Replaces manual X-ray inspection with a highly accurate DenseNet201 model (97.52%).\n• Provides interpretable evidence via Grad-CAM, avoiding the \"black box\" AI problem.\n• Multi-disease fallbacks ensure comprehensive patient safety.\n\n**Stack:** React, FASTAPI, PyTorch, Chart.js",
        quickReplies: ["Compare with other models", "How accurate is it?"]
    },

    // ── General / Small Talk ───────────────────────────────────────────────────
    {
        patterns: ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "namaste", "greetings", "sup"],
        response: "👋 **Hello! I'm CardioBot.**\n\nI am the integrated medical AI assistant for CardioAI. I'm here to answer your questions regarding:\n\n• 🫀 Cardiomegaly & Heart Health\n• 🔬 Our Multi-Disease Classifier\n• 🤖 How our DenseNet201 pipeline works\n• 📤 App Navigation & Result Interpretation\n• 🏥 Discovering local cardiology clinics\n\nHow can I assist you today?",
        quickReplies: ["What is cardiomegaly?", "What is multi-disease?", "How accurate is the AI?", "How to upload X-ray?"]
    },
    {
        patterns: ["thank", "thanks", "great", "awesome", "good", "nice", "perfect", "ok", "cool", "helpful", "wow", "amazing"],
        response: "😊 You're very welcome! I'm glad I could help.\n\nRemember, CardioAI is a clinical support assistant, but early detection through regular doctor checkups is your best defense. Stay heart-healthy!\n\nIf you need anything else, just ask.",
        quickReplies: ["Heart health tips", "Find nearby hospitals"]
    },
    {
        patterns: ["who are you", "what are you", "bot", "ai", "human", "real"],
        response: "🤖 **I am CardioBot!**\n\nI am an AI-driven, offline-first knowledge agent. I have been programmed with technical details regarding the CardioAI DenseNet201 architecture, general cardiology definitions, and emergency protocols.\n\nI do not replace a human doctor, but I can help you navigate this app and understand your X-ray results!",
        quickReplies: ["How does the model work?", "What does the result mean?"]
    },
    {
        patterns: ["bye", "goodbye", "see you", "exit", "quit", "close", "later"],
        response: "👋 **Goodbye!**\n\nStay safe and prioritize your heart health. Feel free to open the chat again whenever you have questions about your X-ray analysis or CardioAI.\n\nTake care! 💚",
        quickReplies: []
    }
];

// ── Fallback ────────────────────────────────────────────────────────────────
const FALLBACK = {
    response: "🤔 I am not quite sure how to answer that.\n\nMy expertise is strictly focused on **Cardiac topics, X-ray AI models, and navigating this application**.\n\nPlease try asking me about:\n• How to interpret your Grad-CAM results\n• The multi-disease classification feature\n• Our DenseNet201 architecture\n• Cardiomegaly symptoms and treatments\n• Heart emergency protocols",
    quickReplies: ["What is cardiomegaly?", "What is multi-disease?", "How accurate is the AI?", "Find nearby hospitals"],
};

/**
 * Gets the best matching response for a user message based on substring matching.
 * Scans the user input against all pattern lists and returns the entry with the highest match score.
 */
export function getResponse(input) {
    const text = input.toLowerCase().trim();
    let best = null;
    let bestScore = 0;

    for (const entry of KB) {
        const score = entry.patterns.reduce((acc, p) => {
            return text.includes(p.toLowerCase()) ? acc + 1 : acc;
        }, 0);

        if (score > bestScore) {
            bestScore = score;
            best = entry;
        }
    }

    return bestScore > 0 ? best : FALLBACK;
}
