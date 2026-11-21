from utils.translator import translate_text

DISEASE_REMEDIES = {
    "Apple___Apple_scab": "Apply fungicides like Captan, Mancozeb, or copper spray. Prune affected shoots and rake/burn fallen diseased leaves to stop overwintering spores.",
    "Apple___Black_rot": "Prune out dead wood, mummified fruits, and cankers. Apply Captan or sulfur-based fungicides during bud break and petal fall.",
    "Apple___Cedar_apple_rust": "Apply Myclobutanil or sulfur fungicide early in the spring. Remove nearby wild juniper/cedar bushes within 500 meters if possible.",
    "Apple___healthy": "Apple tree foliage is healthy! Maintain regular orchard sanitation, balanced NPK fertilization, and proper winter pruning.",
    "Blueberry___healthy": "Blueberry plant is healthy! Maintain acidic soil pH (4.5–5.5), ensure regular organic pine mulch, and use drip irrigation.",
    "Cherry_(including_sour)___Leaf_Spot": "Cherry Leaf Spot (Blumeriella jaapii): Apply Chlorothalonil, Myclobutanil, or copper fungicides at petal fall. Rake up and destroy fallen diseased leaves to eliminate overwintering fungal spores.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply wettable sulfur or potassium bicarbonate sprays early. Prune overcrowded branches to improve sunlight and air circulation.",
    "Cherry_(including_sour)___healthy": "Cherry tree foliage is healthy! Ensure adequate soil moisture during fruit development and monitor for aphids or cherry fruit flies.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply Azoxystrobin, Pyraclostrobin, or Propiconazole fungicides at first sign. Practice crop rotation and plow under crop debris.",
    "Corn_(maize)___Common_rust_": "Apply Mancozeb or Strobilurin fungicides if rust pustules appear before tasseling. Use rust-resistant hybrid seeds for next season.",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply Chlorothalonil or Mancozeb fungicide. Avoid planting corn after corn in the same field; practice two-year crop rotation.",
    "Corn_(maize)___healthy": "Corn crop foliage is healthy! Apply balanced nitrogen top-dressing during knee-high stage and ensure regular irrigation during silking.",
    "Grape___Black_rot": "Spray Mancozeb, Myclobutanil, or Captan before bloom and 2 weeks after. Prune dense foliage and remove mummified grape clusters.",
    "Grape___Esca_(Black_Measles)": "Prune infected vines back to healthy trunk wood. Disinfect pruning shears with 10% bleach solution between cuts. Protect pruning wounds with sealant.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Spray copper oxychloride (0.25%) or Bordeaux mixture (1%) after harvest and during wet periods. Ensure proper canopy aeration.",
    "Grape___healthy": "Grapevine is healthy! Maintain regular canopy management, shoot thinning, and timely irrigation before veraison.",
    "Orange___Haunglongbing_(Citrus_greening)": "Control Asian citrus psyllid insect vectors using systemic insecticides (e.g., Imidacloprid or Thiamethoxam). Remove and destroy severely infected trees to protect the grove.",
    "Peach___Bacterial_spot": "Apply copper-based bactericides at bud swell and petal fall. Avoid excessive nitrogen fertilizer which causes succulent, susceptible tissue.",
    "Peach___Leaf_Curl": "Peach Leaf Curl (Taphrina deformans): Spray copper fungicide or Chlorothalonil during winter dormancy or early spring just before bud swell. Once leaves emerge curled and blistered, sprays cannot cure existing leaves.",
    "Peach___healthy": "Peach tree foliage is healthy! Keep mulch around the tree base, maintain regular watering, and apply dormant copper spray before spring bud break.",
    "Pepper,_bell___Bacterial_spot": "Spray copper hydroxide combined with Mancozeb weekly during rainy periods. Stake plants to keep leaves off damp soil and avoid overhead sprinkler watering.",
    "Pepper,_bell___healthy": "Bell pepper plant is healthy! Ensure adequate calcium to prevent blossom-end rot and maintain steady soil moisture.",
    "Potato___Early_blight": "Apply Mancozeb, Chlorothalonil, or copper fungicides every 7–10 days. Remove lower yellowing leaves and practice 3-year crop rotation.",
    "Potato___Late_blight": "Emergency action: Apply systemic fungicide (Metalaxyl + Mancozeb, Ridomil Gold, or Cymoxanil) immediately. Destroy severely blighted plants to prevent tuber infection.",
    "Potato___healthy": "Potato foliage is healthy! Hill up soil around stems to prevent tuber greening and maintain drip irrigation to keep foliage dry.",
    "Raspberry___healthy": "Raspberry canes are healthy! Prune out old fruited floricanes right after harvest and keep weed-free ground mulch.",
    "Soybean___healthy": "Soybean crop is healthy! Inoculate seeds with Rhizobium for optimal nodulation, monitor for leaf-eating caterpillars, and maintain weed control.",
    "Squash___Powdery_mildew": "Spray neem oil, wettable sulfur, or potassium bicarbonate on both upper and lower leaf surfaces. Ensure wide plant spacing for ventilation.",
    "Strawberry___Leaf_scorch": "Remove and dispose of spotted older leaves after harvest. Apply copper-based fungicide or Captan during damp spring periods.",
    "Strawberry___healthy": "Strawberry plants are healthy! Keep clean straw mulch under runners to prevent fruit soil contact and water early in the morning.",
    "Tomato___Bacterial_spot": "Spray copper hydroxide mixed with Mancozeb. Avoid handling plants when wet, disinfect garden tools, and remove lower diseased leaves.",
    "Tomato___Early_blight": "Remove bottom 12 inches of infected foliage. Spray Chlorothalonil, Mancozeb, or copper fungicide weekly. Mulch soil surface to prevent soil splash.",
    "Tomato___Late_blight": "Urgent: Spray Metalaxyl + Mancozeb (Ridomil Gold) or copper fungicide immediately. Remove and bag severely diseased plants; do not compost blighted leaves.",
    "Tomato___Leaf_Mold": "Improve greenhouse airflow and lower humidity below 85%. Prune dense suckers and apply copper or Chlorothalonil spray.",
    "Tomato___Septoria_leaf_spot": "Clip off infected lower leaves immediately. Apply Chlorothalonil or Mancozeb spray and lay down clean straw or plastic mulch around bases.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray neem oil, insecticidal soap, or Abamectin miticide directly onto undersides of leaves. Wash foliage with strong water spray in dry dusty weather.",
    "Tomato___Target_Spot": "Apply Chlorothalonil or Azoxystrobin fungicide. Improve plant row spacing and weed out wild nightshade weeds.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly vectors using yellow sticky traps and sprays of neem oil or systemic insecticides (Acetamiprid). Remove infected plants to protect healthy ones.",
    "Tomato___Tomato_mosaic_virus": "No chemical cure. Remove and burn infected plants immediately. Wash hands and tools with soap or milk solution before handling other plants. Do not smoke near plants.",
    "Tomato___healthy": "Tomato plant is healthy! Continue regular sucker pruning, stake support, and consistent soil watering to prevent blossom end rot."
}

def get_disease_solution(disease_name, lang="en"):
    """
    Returns an actionable agronomic treatment / solution for the detected plant disease,
    translated into the farmer's preferred language.
    """
    solution_en = DISEASE_REMEDIES.get(
        disease_name,
        "Consult your local agricultural extension officer (Krishi Vigyan Kendra) or apply recommended broad-spectrum protective organic or copper-based bio-fungicide."
    )
    if lang == "en":
        return solution_en
    try:
        return translate_text(solution_en, lang)
    except Exception:
        return solution_en
