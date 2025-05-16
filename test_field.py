// Define Area of Interest (AOI)
var aoi = ee.Geometry.Point([-8.416979, 32.677270]).buffer(500);

Map.centerObject(aoi, 13);

// Load Sentinel-2 imagery and filter by date and AOI
var image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(aoi)
  .filterDate('2019-02-01', '2019-02-28')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .median()
  .clip(aoi);

// Calculate NDVI
var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');

// Calculate BSI (Bare Soil Index)
var bsi = image.expression(
  '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))', {
    'SWIR': image.select('B11'),
    'RED': image.select('B4'),
    'NIR': image.select('B8'),
    'BLUE': image.select('B2')
  }).rename('BSI');

// Apply Canny Edge Detection
var edgesNDVI = ee.Algorithms.CannyEdgeDetector({
  image: ndvi,
  threshold: 0.3,
  sigma: 1
}).rename('NDVI_Edges');

var edgesBSI = ee.Algorithms.CannyEdgeDetector({
  image: bsi,
  threshold: 0.3,
  sigma: 1
}).rename('BSI_Edges');

// Combine edges
var combinedEdges = edgesNDVI.max(edgesBSI).rename('Combined_Edges');

// Dilate the edges to close small gaps
var dilatedEdges = combinedEdges.focal_max({radius: 1, units: 'pixels'}).rename('Dilated_Edges');

// Invert the edges to highlight interior regions (parcels)
var inverted = dilatedEdges.not().rename('Inverted');

// Identify connected regions (potential parcels)
var parcels = inverted.connectedComponents(ee.Kernel.plus(1), 256)
                      .select('labels');

// Vectorize the raster parcels
var parcelVectors = parcels.reduceToVectors({
  geometry: aoi,
  geometryType: 'polygon',
  labelProperty: 'parcel_id',
  eightConnected: false,
  scale: 10,
  maxPixels: 1e8
});

// Add layers to map
Map.addLayer(image, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'RGB');
Map.addLayer(ndvi, {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI');
Map.addLayer(combinedEdges, {min: 0, max: 1}, 'Combined Edges');
Map.addLayer(parcelVectors, {color: 'red'}, 'Predicted Parcel Boundaries');



#########################################


// Define Area of Interest (AOI)
var aoi = ee.Geometry.Point([-8.416979, 32.677270]).buffer(400).bounds();

Map.centerObject(aoi, 13);

// ================== 2. Load Sentinel-2 Imagery ==================
var image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(aoi)
  .filterDate('2018-11-1', '2018-11-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median()
  .clip(aoi);

// ================== 3. Calculate NDVI ==================
var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');

// ================== 4. Calculate BSI ==================
var bsi = image.expression(
  '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))', {
    'SWIR': image.select('B11'),
    'RED': image.select('B4'),
    'NIR': image.select('B8'),
    'BLUE': image.select('B2')
}).rename('BSI');

// ================== 5. Canny Edge Detection ==================
var edgesNDVI = ee.Algorithms.CannyEdgeDetector({
  image: ndvi,
  threshold: 0.1,
  sigma: 1
}).rename('NDVI_Edges');

var edgesBSI = ee.Algorithms.CannyEdgeDetector({
  image: bsi,
  threshold: 0.1,
  sigma: 1
}).rename('BSI_Edges');

// ================== 6. Combine and Dilate Edges ==================
var combinedEdges = edgesNDVI.max(edgesBSI).rename('Combined_Edges');
var dilatedEdges = combinedEdges.focal_max({radius: 1, units: 'pixels'}).rename('Dilated_Edges');

// ================== 7. Invert and Mask for Vegetation ==================
var inverted = dilatedEdges.not().rename('Inverted');
var bsiMask = bsi.gt(0.1);  // bare soil is more reflective
var ndviMask = ndvi.lt(0.2);
var bareMask = ndviMask.and(bsiMask);
var maskedInverted = inverted.updateMask(bareMask);

// ================== 8. Connected Components ==================
var parcels = maskedInverted.connectedComponents(ee.Kernel.plus(1), 256)
                            .select('labels');

// ================== 9. Vectorize and Add Area ==================
var parcelVectors = parcels.reduceToVectors({
  geometry: aoi,
  geometryType: 'polygon',
  labelProperty: 'parcel_id',
  eightConnected: false,
  scale: 10,
  maxPixels: 1e8
});

// Add area attribute
parcelVectors = parcelVectors.map(function(feature) {
  var area = feature.geometry().area(1);  // Add error margin
  var meanNDVI = ndvi.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: feature.geometry(),
    scale: 10,
    maxPixels: 1e8
  }).get('NDVI');
  
  return feature.set({
    'area': area,
    'mean_NDVI': meanNDVI
  });
});


// ================== 10. Filter Small Polygons ==================
parcelVectors = parcelVectors.filter(ee.Filter.gt('area', 500)); // Keep parcels > 500 m²

// ================== 11. Add Layers to Map ==================
Map.addLayer(image, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'RGB');
Map.addLayer(ndvi, {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI');
Map.addLayer(combinedEdges, {min: 0, max: 1}, 'Combined Edges');
Map.addLayer(parcelVectors, {color: 'red'}, 'Predicted Parcel Boundaries');
Map.addLayer(maskedInverted, {min: 0, max: 1, palette: ['red', 'blue']}, 'Masked Inverted');

// ================== 12. Export as Shapefile ==================
// Export.table.toDrive({
//   collection: testCollection,
//   description: 'Test_Export',
//   fileFormat: 'SHP'
// });
