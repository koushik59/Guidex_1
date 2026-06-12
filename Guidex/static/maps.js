/**
 * maps.js — Google Maps & Leaflet Fallback Navigation Logic
 */

let map = null;
let userMarker = null;
let destinationMarker = null;
let routePolyline = null;
let geocoder = null;
let placesService = null;

let isLeafletMode = false;
let leafletMap = null;
let leafletUserMarker = null;
let leafletDestinationMarker = null;
let leafletRoutePolyline = null;

let userLatLng = null; // { lat, lng }
let activeRouteSteps = [];
let activeStepIndex = 0;
let lastGuidanceAt = 0;
let activeDestinationName = '';

// Check if script tag is a placeholder key
const googleScript = Array.from(document.getElementsByTagName('script')).find(s => s.src && s.src.includes('maps/api/js'));
const isPlaceholder = googleScript && (
    googleScript.src.includes('key=AIzaSyDsIVIzXFVHIgnfcT0i88PCGlS9GdAm9pQ') ||
    googleScript.src.includes('key=YOUR_')
);

// Fallback to Leaflet if Google Maps Auth Fails
window.gm_authFailure = function() {
    console.warn("[Maps] Google Maps Authentication Failure. Falling back to Leaflet map.");
    if (!isLeafletMode) {
        destroyGoogleMap();
        initLeafletMap();
    }
};

document.addEventListener('DOMContentLoaded', () => {
    // Initial map setup delay
    setTimeout(() => {
        if (isPlaceholder) {
            console.log("[Maps] Placeholder key detected. Forcing Leaflet map.");
            initLeafletMap();
        } else if (typeof google !== 'undefined' && google.maps) {
            initGoogleMap();
        } else {
            console.log("[Maps] Google Maps not loaded. Forcing Leaflet map.");
            initLeafletMap();
        }
        setupMapControls();
    }, 100);
});

function initGoogleMap() {
    console.log('[MAPS] Initializing Google Maps...');
    isLeafletMode = false;
    const defaultCenter = { lat: 17.3850, lng: 78.4867 }; // Hyderabad

    const mapElement = document.getElementById('map');
    if (!mapElement) return;

    map = new google.maps.Map(mapElement, {
        zoom: 17,
        center: defaultCenter,
        mapTypeId: 'roadmap',
        disableDefaultUI: false,
        zoomControl: true,
        fullscreenControl: true,
        styles: darkMapStyle()
    });

    userMarker = new google.maps.Marker({
        position: defaultCenter,
        map: map,
        title: 'Your Location',
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 10,
            fillColor: '#4CAF50',
            fillOpacity: 1,
            strokeColor: '#ffffff',
            strokeWeight: 2
        }
    });

    geocoder = new google.maps.Geocoder();
    placesService = new google.maps.places.PlacesService(map);
}

function initLeafletMap() {
    console.log('[MAPS] Initializing Leaflet Fallback Map...');
    isLeafletMode = true;
    const defaultCenter = [17.3850, 78.4867];
    const mapElement = document.getElementById('map');
    if (!mapElement) return;

    mapElement.innerHTML = ''; // Clear google map elements if any

    leafletMap = L.map('map').setView(defaultCenter, 17);

    // Dark tiles layout matching modern design aesthetics
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(leafletMap);

    const pulseIcon = L.divIcon({
        className: 'leaflet-user-pulse',
        html: '<div style="width: 14px; height: 14px; background: #4CAF50; border: 2px solid white; border-radius: 50%; box-shadow: 0 0 10px #4CAF50;"></div>',
        iconSize: [14, 14],
        iconAnchor: [7, 7]
    });

    leafletUserMarker = L.marker(defaultCenter, { icon: pulseIcon }).addTo(leafletMap);
}

function destroyGoogleMap() {
    if (map) map = null;
    if (userMarker) {
        userMarker.setMap(null);
        userMarker = null;
    }
    if (destinationMarker) {
        destinationMarker.setMap(null);
        destinationMarker = null;
    }
    if (routePolyline) {
        routePolyline.setMap(null);
        routePolyline = null;
    }
}

function setupMapControls() {
    const centerBtn = document.getElementById('centerMapBtn');
    if (centerBtn) {
        centerBtn.addEventListener('click', () => {
            if (userLatLng) {
                if (isLeafletMode && leafletMap) {
                    leafletMap.setView([userLatLng.lat, userLatLng.lng], 17);
                } else if (map) {
                    map.setCenter(userLatLng);
                    map.setZoom(17);
                }
            }
        });
    }

    const toggleBtn = document.getElementById('toggleMapBtn');
    let isSatellite = false;
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            if (!isLeafletMode && map) {
                isSatellite = !isSatellite;
                map.setMapTypeId(isSatellite ? 'satellite' : 'roadmap');
                toggleBtn.innerHTML = isSatellite ? '<i class="fas fa-map"></i>' : '<i class="fas fa-satellite"></i>';
            } else {
                showToast("Satellite view only available in Google Maps mode");
            }
        });
    }
}

window.updateMapLocation = function(location) {
    userLatLng = location; // { lat, lng }
    if (isLeafletMode) {
        if (leafletMap && leafletUserMarker) {
            leafletUserMarker.setLatLng([location.lat, location.lng]);
            leafletMap.setView([location.lat, location.lng]);
        }
    } else {
        if (map && userMarker) {
            const gLatLng = new google.maps.LatLng(location.lat, location.lng);
            userMarker.setPosition(gLatLng);
            map.setCenter(gLatLng);
        }
    }
    updateLiveGuidance();
};

window.searchMapLocation = async function(locationName) {
    console.log('[MAPS] Navigating to:', locationName);
    activeDestinationName = locationName;

    if (!userLatLng) {
        speak("Your location is not available yet. Please wait.");
        return;
    }

    if (isLeafletMode) {
        geocodeNominatimAndRoute(locationName);
    } else {
        geocodeGoogleAndRoute(locationName);
    }
};

async function geocodeNominatimAndRoute(locationName) {
    const geocodeUrl = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(locationName)}&format=json&limit=1`;
    try {
        const response = await fetch(geocodeUrl, {
            headers: { 'User-Agent': 'GuideX-Wearable-Assistant' }
        });
        const data = await response.json();
        if (data && data.length > 0) {
            const destLat = parseFloat(data[0].lat);
            const destLng = parseFloat(data[0].lon);
            calculateOSRMRoute(destLat, destLng);
        } else {
            showToast(`Could not find destination "${locationName}"`);
            speak(`I could not find the destination ${locationName}`);
        }
    } catch (error) {
        console.error('[MAPS] Nominatim geocode error:', error);
        speak("Geocoding service is currently offline.");
    }
}

async function calculateOSRMRoute(destLat, destLng) {
    const originLat = userLatLng.lat;
    const originLng = userLatLng.lng;
    const osrmUrl = `https://router.projectosrm.org/route/v1/foot/${originLng},${originLat};${destLng},${destLat}?steps=true&geometries=geojson`;

    try {
        const response = await fetch(osrmUrl);
        const data = await response.json();

        if (data.code === 'Ok' && data.routes && data.routes.length > 0) {
            const route = data.routes[0];
            const leg = route.legs[0];

            // Render polyline
            if (leafletRoutePolyline) {
                leafletMap.removeLayer(leafletRoutePolyline);
            }
            const coordinates = route.geometry.coordinates.map(coord => [coord[1], coord[0]]);
            leafletRoutePolyline = L.polyline(coordinates, {
                color: '#2196F3',
                weight: 6,
                opacity: 0.85
            }).addTo(leafletMap);

            // Add marker
            if (leafletDestinationMarker) {
                leafletMap.removeLayer(leafletDestinationMarker);
            }
            const destIcon = L.divIcon({
                className: 'leaflet-dest-marker',
                html: '<div style="width: 14px; height: 14px; background: #f44336; border: 2px solid white; border-radius: 50%; box-shadow: 0 0 10px #f44336;"></div>',
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            });
            leafletDestinationMarker = L.marker([destLat, destLng], { icon: destIcon }).addTo(leafletMap);

            leafletMap.fitBounds(leafletRoutePolyline.getBounds());

            // Save steps
            activeRouteSteps = leg.steps.map(step => {
                let instructionText = step.maneuver.instruction || '';
                if (!instructionText) {
                    const type = step.maneuver.type || 'walk';
                    const modifier = step.maneuver.modifier || '';
                    instructionText = `${type} ${modifier}`.trim();
                }
                return {
                    instructions: instructionText,
                    distance: { text: `${Math.round(step.distance)} m` },
                    end_location: { lat: step.maneuver.location[1], lng: step.maneuver.location[0] }
                };
            });

            activeStepIndex = 0;
            lastGuidanceAt = 0;

            const distanceKm = (route.distance / 1000).toFixed(1);
            const durationMin = Math.round(route.duration / 60);

            const navResults = document.getElementById('navResults');
            if (navResults) {
                navResults.innerHTML = `<div class="nav-result">
                    <strong>Destination:</strong> ${activeDestinationName}<br>
                    <strong>Distance:</strong> ${distanceKm} km<br>
                    <strong>Estimated Time:</strong> ${durationMin} mins
                </div>`;
            }

            speak(`Route to ${activeDestinationName} calculated. Distance is ${distanceKm} kilometers.`);
            setTimeout(() => speakCurrentStep('First instruction'), 1500);

        } else {
            speak("Could not calculate walking path.");
        }
    } catch (error) {
        console.error('[MAPS] OSRM route error:', error);
        speak("Routing service is offline.");
    }
}

function geocodeGoogleAndRoute(locationName) {
    geocoder.geocode({ address: locationName }, (results, status) => {
        if (status === 'OK' && results.length > 0) {
            const destLoc = results[0].geometry.location;

            if (destinationMarker) destinationMarker.setMap(null);
            destinationMarker = new google.maps.Marker({
                position: destLoc,
                map: map,
                title: locationName,
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 10,
                    fillColor: '#f44336',
                    fillOpacity: 1,
                    strokeColor: '#ffffff',
                    strokeWeight: 2
                }
            });

            calculateGoogleRoute(destLoc);
        } else {
            speak(`Could not find ${locationName}`);
        }
    });
}

function calculateGoogleRoute(destinationLatLng) {
    const directionsService = new google.maps.DirectionsService();
    const origin = new google.maps.LatLng(userLatLng.lat, userLatLng.lng);

    directionsService.route(
        {
            origin: origin,
            destination: destinationLatLng,
            travelMode: 'WALKING'
        },
        (response, status) => {
            if (status === 'OK') {
                const route = response.routes[0];
                const polylinePoints = [];

                route.legs.forEach(leg => {
                    leg.steps.forEach(step => {
                        step.path.forEach(point => polylinePoints.push(point));
                    });
                });

                if (routePolyline) routePolyline.setMap(null);
                routePolyline = new google.maps.Polyline({
                    path: polylinePoints,
                    geodesic: true,
                    strokeColor: '#2196F3',
                    strokeOpacity: 0.8,
                    strokeWeight: 6,
                    map: map
                });

                // Set steps
                activeRouteSteps = route.legs[0].steps.map(step => ({
                    instructions: step.instructions,
                    distance: { text: step.distance.text },
                    end_location: { lat: step.end_location.lat(), lng: step.end_location.lng() }
                }));

                activeStepIndex = 0;
                lastGuidanceAt = 0;

                const bounds = new google.maps.LatLngBounds();
                bounds.extend(origin);
                bounds.extend(destinationLatLng);
                map.fitBounds(bounds);

                let totalDistance = 0;
                let totalDuration = 0;
                route.legs.forEach(leg => {
                    totalDistance += leg.distance.value;
                    totalDuration += leg.duration.value;
                });

                const distanceKm = (totalDistance / 1000).toFixed(1);
                const durationMin = Math.round(totalDuration / 60);

                const navResults = document.getElementById('navResults');
                if (navResults) {
                    navResults.innerHTML = `<div class="nav-result">
                        <strong>Destination:</strong> ${activeDestinationName}<br>
                        <strong>Distance:</strong> ${distanceKm} km<br>
                        <strong>Estimated Time:</strong> ${durationMin} mins
                    </div>`;
                }

                speak(`Route to ${activeDestinationName} calculated. Distance is ${distanceKm} kilometers.`);
                setTimeout(() => speakCurrentStep('First instruction'), 1500);
            } else {
                console.warn("[MAPS] Google Directions failed, falling back to OSRM.");
                calculateOSRMRoute(destinationLatLng.lat(), destinationLatLng.lng());
            }
        }
    );
}

window.clearRoute = function() {
    if (isLeafletMode) {
        if (leafletRoutePolyline) {
            leafletMap.removeLayer(leafletRoutePolyline);
            leafletRoutePolyline = null;
        }
        if (leafletDestinationMarker) {
            leafletMap.removeLayer(leafletDestinationMarker);
            leafletDestinationMarker = null;
        }
    } else {
        if (routePolyline) {
            routePolyline.setMap(null);
            routePolyline = null;
        }
        if (destinationMarker) {
            destinationMarker.setMap(null);
            destinationMarker = null;
        }
    }
    const navResults = document.getElementById('navResults');
    if (navResults) navResults.innerHTML = '';
    document.getElementById('locationInput').value = '';
    activeRouteSteps = [];
    activeStepIndex = 0;
    activeDestinationName = '';
    lastGuidanceAt = 0;
};

function stripInstruction(html) {
    const temp = document.createElement('div');
    temp.innerHTML = html || '';
    return temp.textContent || temp.innerText || '';
}

function getDistanceMeters(fromLoc, toLoc) {
    if (!fromLoc || !toLoc) return Infinity;
    const lat1 = fromLoc.lat * Math.PI / 180;
    const lat2 = toLoc.lat * Math.PI / 180;
    const dLat = lat2 - lat1;
    const dLng = (toLoc.lng - fromLoc.lng) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
        Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
    return 6371000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

async function speakCurrentStep(prefix = 'Next direction') {
    if (!activeRouteSteps.length || activeStepIndex >= activeRouteSteps.length) {
        return;
    }

    const step = activeRouteSteps[activeStepIndex];
    let instruction = stripInstruction(step.instructions);
    const distance = step.distance ? step.distance.text : '';

    const lang = document.getElementById('languageSelect').value;
    if (lang !== 'en') {
        try {
            const res = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: instruction, target_lang: lang })
            });
            const data = await res.json();
            if (data.translated) {
                instruction = data.translated;
            }
        } catch (e) {
            console.warn("Translation failed for instruction:", e);
        }
    }

    let localizedPrefix = prefix;
    if (lang === 'hi') {
        if (prefix === 'First instruction') localizedPrefix = 'पहला निर्देश';
        else if (prefix === 'Next instruction') localizedPrefix = 'अगला निर्देश';
        else if (prefix === 'Now') localizedPrefix = 'अब';
        else if (prefix === 'Continue') localizedPrefix = 'आगे बढ़ें';
        speak(`${localizedPrefix}. ${distance} में, ${instruction}.`);
    } else if (lang === 'te') {
        if (prefix === 'First instruction') localizedPrefix = 'మొదటి సూచన';
        else if (prefix === 'Next instruction') localizedPrefix = 'తరువాతి సూచన';
        else if (prefix === 'Now') localizedPrefix = 'ఇప్పుడు';
        else if (prefix === 'Continue') localizedPrefix = 'కొనసాగించండి';
        speak(`${localizedPrefix}. ${distance} లో, ${instruction}.`);
    } else {
        speak(`${localizedPrefix}. In ${distance}, ${instruction}.`);
    }

    lastGuidanceAt = Date.now();
}

function updateLiveGuidance() {
    if (!activeRouteSteps.length || !userLatLng || activeStepIndex >= activeRouteSteps.length) {
        return;
    }

    const step = activeRouteSteps[activeStepIndex];
    const distanceToStepEnd = getDistanceMeters(userLatLng, step.end_location);
    const now = Date.now();

    if (distanceToStepEnd < 30 && activeStepIndex < activeRouteSteps.length - 1) {
        activeStepIndex += 1;
        speakCurrentStep('Now');
        return;
    }

    if (now - lastGuidanceAt > 40000) {
        speakCurrentStep('Continue');
    }
}

function speakMapLocation() {
    if (userLatLng) {
        const lat = userLatLng.lat;
        const lng = userLatLng.lng;
        const lang = document.getElementById('languageSelect').value;
        if (lang === 'hi') {
            speak(`आपकी वर्तमान स्थिति अक्षांश ${lat.toFixed(5)}, देशांतर ${lng.toFixed(5)} है।`);
        } else if (lang === 'te') {
            speak(`మీ ప్రస్తుత స్థానం అక్షాంశం ${lat.toFixed(5)}, రేఖాంశం ${lng.toFixed(5)}.`);
        } else {
            speak(`Your current coordinates are latitude ${lat.toFixed(5)}, longitude ${lng.toFixed(5)}.`);
        }
    } else {
        speak('Location details not loaded yet.');
    }
}

function darkMapStyle() {
    return [
        { elementType: 'geometry', stylers: [{ color: '#0b0b14' }] },
        { elementType: 'labels.text.stroke', stylers: [{ color: '#0b0b14' }] },
        { elementType: 'labels.text.fill', stylers: [{ color: '#7c7c94' }] },
        { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#141424' }] },
        { featureType: 'road', elementType: 'geometry.stroke', stylers: [{ color: '#202038' }] },
        { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#8b8ba8' }] },
        { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#252654' }] },
        { featureType: 'road.highway', elementType: 'geometry.stroke', stylers: [{ color: '#313270' }] },
        { featureType: 'road.highway', elementType: 'labels.text.fill', stylers: [{ color: '#a07bf0' }] },
        { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#07080e' }] },
        { featureType: 'water', elementType: 'labels.text.fill', stylers: [{ color: '#4b5066' }] },
        { featureType: 'poi', elementType: 'labels.text.fill', stylers: [{ color: '#a855f7' }] },
        { featureType: 'poi.park', elementType: 'geometry', stylers: [{ color: '#122022' }] },
        { featureType: 'poi.park', elementType: 'labels.text.fill', stylers: [{ color: '#55825f' }] },
        { featureType: 'transit', elementType: 'geometry', stylers: [{ color: '#1c1c38' }] },
        { featureType: 'transit.station', elementType: 'labels.text.fill', stylers: [{ color: '#a855f7' }] },
        { featureType: 'administrative', elementType: 'geometry.stroke', stylers: [{ color: '#3b3b68' }] },
        { featureType: 'administrative.land_parcel', elementType: 'labels.text.fill', stylers: [{ color: '#5b6b8f' }] },
        { featureType: 'administrative.province', elementType: 'geometry.stroke', stylers: [{ color: '#3b3b68' }] },
    ];
}