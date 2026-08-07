"""
Static knowledge base describing each of the 38 classes the Plantix model
was trained on. Used to render friendly, actionable information alongside
a model prediction (crop, condition, symptoms, and suggested care).

NOTE: This is general horticultural guidance for informational purposes.
Always confirm severe or spreading infections with a local agronomist /
extension office before applying chemical treatments.
"""

DISEASE_INFO = {
    "Apple Black Rot": {
        "crop": "Apple", "condition": "Black Rot", "healthy": False, "severity": "High",
        "description": "A fungal disease (Botryosphaeria obtusa) causing leaf spots, fruit rot and cankers on branches.",
        "symptoms": ["Purple-bordered leaf spots", "Concentric rings on fruit", "Sunken bark cankers"],
        "treatment": ["Prune and destroy infected wood/mummified fruit", "Apply a labeled fungicide during the growing season", "Improve air circulation by thinning the canopy"],
    },
    "Apple Cedar Apple Rust": {
        "crop": "Apple", "condition": "Cedar Apple Rust", "healthy": False, "severity": "Medium",
        "description": "A fungal disease requiring both apple and cedar/juniper hosts, producing bright orange leaf spots.",
        "symptoms": ["Yellow-orange spots on upper leaf surface", "Orange tube-like structures on leaf underside", "Premature leaf drop"],
        "treatment": ["Remove nearby juniper/cedar hosts if possible", "Apply protectant fungicide from pink bud to mid-summer", "Choose rust-resistant apple varieties"],
    },
    "Apple Healthy": {
        "crop": "Apple", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Apple Scab": {
        "crop": "Apple", "condition": "Scab", "healthy": False, "severity": "High",
        "description": "A common fungal disease (Venturia inaequalis) that produces olive-green to black scabby lesions.",
        "symptoms": ["Olive-green velvety spots on leaves", "Scabby, corky lesions on fruit", "Leaf distortion and early drop"],
        "treatment": ["Rake and destroy fallen leaves each autumn", "Apply fungicide starting at bud break", "Plant scab-resistant cultivars"],
    },
    "Blueberry Healthy": {
        "crop": "Blueberry", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain acidic, well-drained soil", "Continue regular monitoring"],
    },
    "Cherry(including sour) Healthy": {
        "crop": "Cherry", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Cherry(including sour) Powdery Mildew": {
        "crop": "Cherry", "condition": "Powdery Mildew", "healthy": False, "severity": "Medium",
        "description": "A fungal disease (Podosphaera clandestina) that coats leaves and shoots with white powdery growth.",
        "symptoms": ["White powdery patches on leaves/shoots", "Leaf curling and stunted growth", "Reduced fruit quality"],
        "treatment": ["Improve air circulation via pruning", "Apply sulfur-based or labeled fungicide", "Avoid excess nitrogen fertilization"],
    },
    "Corn(maize) Cercospora Leaf Spot Gray Leaf Spot": {
        "crop": "Corn (Maize)", "condition": "Gray Leaf Spot", "healthy": False, "severity": "High",
        "description": "A fungal disease (Cercospora zeae-maydis) producing rectangular gray-to-tan lesions between leaf veins.",
        "symptoms": ["Rectangular tan/gray lesions", "Lesions run parallel to leaf veins", "Lower leaves affected first"],
        "treatment": ["Rotate crops and till residue", "Plant resistant hybrids", "Apply foliar fungicide if disease pressure is high"],
    },
    "Corn(maize) Common Rust": {
        "crop": "Corn (Maize)", "condition": "Common Rust", "healthy": False, "severity": "Medium",
        "description": "A fungal disease (Puccinia sorghi) that forms reddish-brown pustules on both leaf surfaces.",
        "symptoms": ["Small cinnamon-brown pustules", "Pustules on both leaf sides", "Leaves may yellow with heavy infection"],
        "treatment": ["Plant rust-resistant hybrids", "Apply fungicide if infection appears before tasseling", "Avoid overhead irrigation late in the day"],
    },
    "Corn(maize) Healthy": {
        "crop": "Corn (Maize)", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Corn(maize) Northern Leaf Blight": {
        "crop": "Corn (Maize)", "condition": "Northern Leaf Blight", "healthy": False, "severity": "High",
        "description": "A fungal disease (Exserohilum turcicum) causing long, elliptical gray-green lesions.",
        "symptoms": ["Cigar-shaped gray-green lesions", "Lesions enlarge and merge", "Premature leaf death in severe cases"],
        "treatment": ["Rotate with non-host crops", "Plant resistant hybrids", "Apply fungicide at first sign of disease"],
    },
    "Grape Black Rot": {
        "crop": "Grape", "condition": "Black Rot", "healthy": False, "severity": "High",
        "description": "A fungal disease (Guignardia bidwellii) that shrivels berries into hard black 'mummies'.",
        "symptoms": ["Circular brown leaf spots with dark borders", "Black shriveled, mummified berries", "Small black fruiting bodies on lesions"],
        "treatment": ["Remove mummified berries and infected debris", "Apply fungicide from bud break through veraison", "Improve canopy airflow with proper pruning"],
    },
    "Grape Esca (Black Measles)": {
        "crop": "Grape", "condition": "Esca (Black Measles)", "healthy": False, "severity": "High",
        "description": "A complex fungal trunk disease causing tiger-stripe leaf discoloration and berry spotting.",
        "symptoms": ["Tiger-stripe interveinal leaf discoloration", "Dark spots on berries", "Sudden vine collapse in severe cases"],
        "treatment": ["Prune out and destroy infected wood", "Avoid pruning wounds during wet weather", "Apply trunk wound protectants after pruning"],
    },
    "Grape Healthy": {
        "crop": "Grape", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Grape Leaf Blight (Isariopsis Leaf Spot)": {
        "crop": "Grape", "condition": "Leaf Blight (Isariopsis Leaf Spot)", "healthy": False, "severity": "Medium",
        "description": "A fungal disease producing angular brown spots that can lead to early defoliation.",
        "symptoms": ["Angular reddish-brown leaf spots", "Yellow halo around lesions", "Premature leaf drop"],
        "treatment": ["Remove and destroy fallen leaves", "Apply copper-based or labeled fungicide", "Ensure good vineyard air circulation"],
    },
    "Orange Haunglongbing (Citrus greening)": {
        "crop": "Orange", "condition": "Huanglongbing (Citrus Greening)", "healthy": False, "severity": "Very High",
        "description": "A serious bacterial disease spread by the Asian citrus psyllid; currently incurable and often fatal to trees.",
        "symptoms": ["Blotchy, asymmetric yellow mottling on leaves", "Small, lopsided, bitter fruit", "Twig dieback and gradual tree decline"],
        "treatment": ["Remove and destroy infected trees to limit spread", "Control the Asian citrus psyllid vector", "Use certified disease-free nursery stock", "Report suspected cases to local agricultural authorities"],
    },
    "Peach Bacterial Spot": {
        "crop": "Peach", "condition": "Bacterial Spot", "healthy": False, "severity": "Medium",
        "description": "A bacterial disease (Xanthomonas campestris) causing small dark spots and fruit blemishes.",
        "symptoms": ["Small angular dark leaf spots", "Spots may fall out, leaving a 'shot-hole' look", "Sunken pitted lesions on fruit"],
        "treatment": ["Plant resistant peach varieties", "Apply copper-based bactericide during dormancy", "Avoid overhead irrigation"],
    },
    "Peach Healthy": {
        "crop": "Peach", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Pepper Bell Bacterial Spot": {
        "crop": "Bell Pepper", "condition": "Bacterial Spot", "healthy": False, "severity": "Medium",
        "description": "A bacterial disease (Xanthomonas spp.) that creates small water-soaked spots on leaves and fruit.",
        "symptoms": ["Small water-soaked leaf spots turning brown", "Raised scabby spots on fruit", "Leaf yellowing and drop"],
        "treatment": ["Use certified disease-free seed/transplants", "Apply copper-based bactericide", "Avoid working in fields when foliage is wet"],
    },
    "Pepper Bell Healthy": {
        "crop": "Bell Pepper", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Potato Early Blight": {
        "crop": "Potato", "condition": "Early Blight", "healthy": False, "severity": "Medium",
        "description": "A fungal disease (Alternaria solani) producing dark concentric 'target-spot' lesions.",
        "symptoms": ["Dark brown spots with concentric rings", "Yellowing tissue around lesions", "Lower, older leaves affected first"],
        "treatment": ["Rotate crops on a 2-3 year cycle", "Apply fungicide at first symptom onset", "Maintain balanced soil fertility"],
    },
    "Potato Healthy": {
        "crop": "Potato", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Potato Late Blight": {
        "crop": "Potato", "condition": "Late Blight", "healthy": False, "severity": "Very High",
        "description": "A fast-spreading water mold disease (Phytophthora infestans) — the cause of the historic Irish famine.",
        "symptoms": ["Dark, water-soaked lesions with pale green borders", "White fungal growth on leaf undersides in humid weather", "Rapid tissue collapse and rot"],
        "treatment": ["Destroy infected plants immediately to stop spread", "Apply protectant fungicide before rain events", "Avoid overhead irrigation and improve drainage"],
    },
    "Raspberry Healthy": {
        "crop": "Raspberry", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Soybean Healthy": {
        "crop": "Soybean", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Squash Powdery Mildew": {
        "crop": "Squash", "condition": "Powdery Mildew", "healthy": False, "severity": "Medium",
        "description": "A common fungal disease coating leaves in white powder, reducing photosynthesis.",
        "symptoms": ["White powdery spots on leaves and stems", "Leaves yellow and curl", "Reduced fruit yield"],
        "treatment": ["Space plants for airflow and full sun", "Apply sulfur, potassium bicarbonate, or a labeled fungicide", "Remove severely affected leaves"],
    },
    "Strawberry Healthy": {
        "crop": "Strawberry", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Strawberry Leaf Scorch": {
        "crop": "Strawberry", "condition": "Leaf Scorch", "healthy": False, "severity": "Medium",
        "description": "A fungal disease (Diplocarpon earlianum) producing purple blotches that can merge into scorched patches.",
        "symptoms": ["Small purple spots on leaves", "Spots merge into large scorched-looking blotches", "Reduced plant vigor"],
        "treatment": ["Remove infected leaves after harvest", "Apply fungicide during the growing season", "Avoid overhead watering, favor drip irrigation"],
    },
    "Tomato Bacterial Spot": {
        "crop": "Tomato", "condition": "Bacterial Spot", "healthy": False, "severity": "Medium",
        "description": "A bacterial disease (Xanthomonas spp.) producing small dark, greasy leaf and fruit spots.",
        "symptoms": ["Small dark water-soaked spots on leaves", "Scabby raised spots on fruit", "Leaf yellowing and drop"],
        "treatment": ["Use disease-free seed/transplants", "Apply copper-based bactericide", "Rotate crops and avoid working wet plants"],
    },
    "Tomato Early Blight": {
        "crop": "Tomato", "condition": "Early Blight", "healthy": False, "severity": "Medium",
        "description": "A fungal disease (Alternaria solani) causing dark target-like spots, usually on older leaves first.",
        "symptoms": ["Brown spots with concentric rings", "Yellowing around lesions", "Lower leaves affected first"],
        "treatment": ["Remove and destroy infected lower leaves", "Apply fungicide preventatively", "Mulch to reduce soil splash onto leaves"],
    },
    "Tomato Healthy": {
        "crop": "Tomato", "condition": "Healthy", "healthy": True, "severity": "None",
        "description": "The leaf shows no visible signs of disease or nutrient stress.",
        "symptoms": ["Uniform green color", "No spots, lesions or curling"],
        "treatment": ["Maintain regular watering and fertilization", "Monitor periodically for early signs of pests or disease"],
    },
    "Tomato Late Blight": {
        "crop": "Tomato", "condition": "Late Blight", "healthy": False, "severity": "Very High",
        "description": "A fast-moving water mold disease (Phytophthora infestans) that can destroy a crop within days.",
        "symptoms": ["Large water-soaked, greasy-looking lesions", "White fuzzy growth on leaf undersides", "Rapid browning and plant collapse"],
        "treatment": ["Remove and destroy infected plants immediately", "Apply protectant fungicide ahead of cool, wet weather", "Avoid overhead irrigation"],
    },
    "Tomato Leaf Mold": {
        "crop": "Tomato", "condition": "Leaf Mold", "healthy": False, "severity": "Medium",
        "description": "A fungal disease (Passalora fulva) common in humid greenhouse conditions.",
        "symptoms": ["Pale yellow spots on upper leaf surface", "Olive-green to gray fuzzy mold underneath", "Leaves curl and wither"],
        "treatment": ["Improve ventilation and reduce humidity", "Space plants to increase airflow", "Apply fungicide labeled for leaf mold"],
    },
    "Tomato Mosaic Virus": {
        "crop": "Tomato", "condition": "Mosaic Virus", "healthy": False, "severity": "High",
        "description": "A viral disease causing mottled, distorted foliage and reduced yield; no chemical cure exists.",
        "symptoms": ["Light and dark green mottled pattern", "Leaf curling, fern-like distortion", "Stunted plant growth"],
        "treatment": ["Remove and destroy infected plants", "Disinfect tools between plants", "Wash hands after handling tobacco products before touching plants", "Use resistant varieties where available"],
    },
    "Tomato Septoria Leaf Spot": {
        "crop": "Tomato", "condition": "Septoria Leaf Spot", "healthy": False, "severity": "Medium",
        "description": "A fungal disease (Septoria lycopersici) producing many small circular spots with dark borders.",
        "symptoms": ["Small circular spots with gray centers and dark edges", "Tiny black fruiting bodies in spot centers", "Lower leaves yellow and drop first"],
        "treatment": ["Remove infected lower leaves promptly", "Apply fungicide at first sign of spots", "Mulch and avoid wetting foliage"],
    },
    "Tomato Spider Mites Two-spotted Spider Mite": {
        "crop": "Tomato", "condition": "Two-Spotted Spider Mite", "healthy": False, "severity": "Medium",
        "description": "A tiny sap-sucking pest that thrives in hot, dry conditions and causes stippled, bronzed foliage.",
        "symptoms": ["Fine yellow/white stippling on leaves", "Fine webbing on leaf undersides", "Leaves turn bronze and dry out"],
        "treatment": ["Spray leaves (especially undersides) with water to dislodge mites", "Apply insecticidal soap or miticide", "Introduce predatory mites as biological control"],
    },
    "Tomato Target Spot": {
        "crop": "Tomato", "condition": "Target Spot", "healthy": False, "severity": "Medium",
        "description": "A fungal disease (Corynespora cassiicola) producing brown lesions with concentric target-like rings.",
        "symptoms": ["Brown lesions with concentric rings", "Lesions on leaves, stems and fruit", "Leaf yellowing and premature drop"],
        "treatment": ["Improve air circulation via pruning/staking", "Apply fungicide at early disease onset", "Rotate crops with non-host plants"],
    },
    "Tomato Yellow Leaf Curl Virus": {
        "crop": "Tomato", "condition": "Yellow Leaf Curl Virus", "healthy": False, "severity": "Very High",
        "description": "A whitefly-transmitted viral disease causing severe stunting and yield loss; no chemical cure exists.",
        "symptoms": ["Upward curling, yellowing leaves", "Severe stunting of new growth", "Flower drop and poor fruit set"],
        "treatment": ["Control whitefly populations with sticky traps/insecticide", "Remove and destroy infected plants", "Use resistant varieties and reflective mulches"],
    },
}

CLASS_NAMES = list(DISEASE_INFO.keys())