import ee
import geemap

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='certain-catcher-430110-v2')

# 1. Define AOI
aoi = ee.Geometry.Polygon([[[7.3, 10.4], [7.3, 10.6], [7.5, 10.6], [7.5, 10.4]]]).simplify(100)

# 2. Extract cropland parcels
land_cover = ee.ImageCollection("ESA/WorldCover/v100").filterBounds(aoi).first().clip(aoi)
parcels = land_cover.eq(40).selfMask().reduceToVectors(
    geometry=aoi,
    scale=10,
    geometryType='polygon',
    labelProperty='is_parcel',
    maxPixels=1e10
).map(lambda f: f.set('is_parcel', 1))

# 3. Negative sampling with improved buffer distance
parcel_geom = parcels.geometry().buffer(30)  # Increased buffer to 30m
non_parcel_area = aoi.difference(parcel_geom, 10)

non_parcels = ee.FeatureCollection(
    ee.Algorithms.If(
        non_parcel_area.area().gt(1e6),
        ee.FeatureCollection.randomPoints(
            region=non_parcel_area,
            points=500,
            seed=42
        ).map(lambda f: f.buffer(30).set('is_parcel', 0)),
        ee.FeatureCollection([])
    )
)

# 4. Load Sentinel-2 with improved filtering
s2 = (ee.ImageCollection('COPERNICUS/S2_SR')
      .filterDate('2022-01-01', '2022-12-31')  # Extended to full year
      .filterBounds(aoi)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))  # Stricter cloud filter
      .median()
      .clip(aoi))

# 5. Feature engineering (modified texture calculation)
def addIndices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    bsi = image.expression(
        '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))',
        {'B11': image.select('B11'), 'B4': image.select('B4'), 
         'B8': image.select('B8'), 'B2': image.select('B2')}
    ).rename('BSI')
    evi = image.expression(
        '2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))',
        {'B8': image.select('B8'), 'B4': image.select('B4'), 'B2': image.select('B2')}
    ).rename('EVI')
    savi = image.expression(
        '((B8 - B4) / (B8 + B4 + 0.5)) * 1.5',
        {'B8': image.select('B8'), 'B4': image.select('B4')}
    ).rename('SAVI')
    
    # Convert B8 to integer for texture calculation
    b8_int = image.select('B8').multiply(10000).toInt()
    texture = b8_int.glcmTexture(size=3).select('B8_contrast').rename('B8_contrast')
    
    return image.addBands([ndvi, ndwi, bsi, evi, savi, texture])


features = addIndices(s2)

# 6. Combine training data with validation split
training_data = parcels.merge(non_parcels)

# Add random column for splitting
training_data = training_data.randomColumn('random')

# Split 70% training, 30% validation
training_set = training_data.filter(ee.Filter.lt('random', 0.7))
validation_set = training_data.filter(ee.Filter.gte('random', 0.7))

# Sample training data
training_samples = features.sampleRegions(
    collection=training_set,
    properties=['is_parcel'],
    scale=10,
    tileScale=8
)

# 7. Train classifier if enough samples
if training_samples.size().getInfo() > 100:
    classifier = ee.Classifier.smileRandomForest(50).train(
        features=training_samples,
        classProperty='is_parcel',
        inputProperties=features.bandNames()
    )
    
    # Validate classifier
    validation_samples = features.sampleRegions(
        collection=validation_set,
        properties=['is_parcel'],
        scale=10,
        tileScale=8
    )
    validated = validation_samples.classify(classifier)
    
    # Calculate accuracy
    error_matrix = validated.errorMatrix('is_parcel', 'classification')
    print('Validation accuracy:', error_matrix.accuracy().getInfo())
    print('Error matrix:', error_matrix.getInfo())
    
    # 8. Classify entire image
    predicted = features.classify(classifier)
else:
    print("Not enough training samples")
    predicted = ee.Image(0).rename('classification')

# 9. Extract parcel boundaries with improved filtering
boundaries = (predicted.gt(0.5)
    .focalMode(radius=2, units='pixels')
    .reduceToVectors(
        geometry=aoi,
        scale=10,
        geometryType='polygon',
        maxPixels=1e10
    )
    .filter(ee.Filter.gt('area', 1000)))  # Increased minimum area to 1000 sqm

# Visualization
Map = geemap.Map(center=[10.5, 7.4], zoom=12)
Map.addLayer(s2, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}, 'Sentinel-2 RGB')
Map.addLayer(predicted.selfMask(), {'min': 0, 'max': 1, 'palette': ['white', 'green']}, 'Predictions')
Map.addLayer(boundaries, {'color': 'red'}, 'Field Boundaries')
Map.addLayer(parcels, {'color': 'yellow'}, 'Training Parcels')
Map.addLayer(non_parcels, {'color': 'blue'}, 'Negative Samples')

# Add layer controls
Map.addLayerControl()

# Display the map
Map