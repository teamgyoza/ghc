/**
 * @fileoverview Contains visualization script for Google Hash Contest app.
 * Requires jQuery, jQuery UI and google.maps API.
 */

(function($) {

  /**
   * StreetView submission visualizer.
   *
   * Options:
   *  - colors: a list of colors used to draw paths (will cycle).
   *  - duration: duration of animation in milliseconds.
   *  - sleep: animation step sleep period in milliseconds.
   *  - zoom: map zoom level.
   *  - legend: element, if provided will be used for interactive map legend.
   *  - slider: div to display a slider.
   *
   * Itineraries JSON format: array of car itineraries, where each itinerary is
   * a list of stops, and a stop is [ [ Lat, Lng ], Time ].
   *
   */
  $.fn.visualizeStreets = function(streets, options, slider) {
    var self;
    $(this).each(function() {
        self = this;
    });
    var defaults = {
      colors: ['#1040E0', '#EE1729', '#07A915', '#FAAC00',
               '#00DDE0', '#C828EE', '#36FB00', '#A0522D'],
      duration: 60000,
      sleep: 100,
      zoom: 13,
      maxZoom: 17,
      strokeWeight: 2.5,
      legend: null,
      slider: null,
      streetColor: "#000000"
    };
    var settings = $.extend({}, defaults, options);

    // Center la ville des lumières.
    var centerLat = 48.86;
    var centerLng = 2.337;

    var map = new google.maps.Map(self, {
        center: new google.maps.LatLng(centerLat, centerLng),
        zoom: settings.zoom,
        mapTypeId: google.maps.MapTypeId.ROADMAP
    });

    //draw streets
    _.each(streets.edges, function(edge) {
        var path = new google.maps.Polyline({
            geodesic: true,
            strokeOpacity: 0.8,
            strokeColor: settings.streetColor,
            strokeWeight: settings.strokeWeight - 0.5,
            zIndex: 100
        });
        path.setMap(map);
        path.getPath().push(new google.maps.LatLng(edge.start.x, edge.start.y));
        path.getPath().push(new google.maps.LatLng(edge.stop.x, edge.stop.y));
        path.addListener('click', function(e) {
            var infowindow = new google.maps.InfoWindow({
                content: '<div id="content"><ul>' +
                    '<li>edge.idx : ' + edge.idx + '</li>' +
                    '<li>edge.distance : ' + edge.distance + '</li>' +
                    '<li>edge.cost : ' + edge.cost + '</li>' +
                    '<li>edge.start.idx : ' + edge.start.idx + '</li>' +
                    '<li>edge.stop.idx : ' + edge.stop.idx + '</li>' +
                    '<li>edge.stop.bothways : ' + edge.bothways + '</li></ul></div>'
            });
            infowindow.setPosition(e.latLng);
            infowindow.open(map);
        });
      });
    function drawItineraries(itineraries) {
        // Find the max end time from all itineraries and initialize Polylines.
        var maxTime = 0;
        var carPaths = [];
        var lasts = []; // last point of the itinerary drawn in the prev step;
        for (var i = 0; i < itineraries.length; ++i) {
            var itinerary = itineraries[i];
            if (itinerary.length > 0) {
                maxTime = Math.max(maxTime, itinerary[itinerary.length - 1][1]);
            }
            var path = new google.maps.Polyline({
                geodesic: true,
                strokeColor: settings.colors[i % settings.colors.length],
                strokeWeight: settings.strokeWeight,
                zIndex: 300
            });
            path.setMap(map);
            carPaths.push(path);
            lasts.push(-1);
        }
        var timeStep = maxTime / (settings.duration / settings.sleep);
        var time = 0;
        var targetTime = 0;
        var userInteracting = false;

        var slider = settings.slider;
        console.log(maxTime);
        if (slider) {
            slider.slider({
            min: 0,
            max: maxTime,
            change: function(event, ui) {
                if (ui.value < maxTime) {
                targetTime = ui.value;
                }
            },
            start: function(event, ui) { userInteracting = true; },
            stop: function(event, ui) { userInteracting = false; }
            });
        }
        // Paint all segments.
        function step() {
            if (targetTime < time) {
            for (var i = 0; i < itineraries.length; ++i) {
                carPaths[i].getPath().clear();
                time = 0;
                lasts[i] = -1;
            }
            }
            if (targetTime <= maxTime) {
                targetTime += timeStep;
                for (var i = 0; i < itineraries.length; ++i) {
                    var itinerary = itineraries[i];
                    for (var j = lasts[i] + 1; j < itinerary.length &&
                        itinerary[j][1] < targetTime; ++j) {
                    var lat = itinerary[j][0][0];
                    var lng = itinerary[j][0][1];
                    carPaths[i].getPath().push(new google.maps.LatLng(lat, lng));
                    lasts[i] = j;
                    }
                }
                time = targetTime;
                }
            if (slider && !userInteracting) {
                slider.slider('option', 'value', time);
            }
            setTimeout(step, settings.sleep);
        };
        // Setup game legend.
        if (settings.legend) {
            settings.legend.html("");
            var carList = $('<ul/>').appendTo(settings.legend);
            $.each(carPaths, function(index, carPath) {
            var carItem = $('<li>Car ' + (index + 1) + '</li>')
                .css('color', settings.colors[index % settings.colors.length])
                .appendTo(carList)
                .click(function() {
                    if (carPath.getMap()) {
                    carPath.setMap(null);
                    $(self).css('text-decoration', 'line-through');
                    } else {
                    carPath.setMap(map);
                    $(self).css('text-decoration', 'none');
                    }
                });
            });
          }
        setTimeout(step, settings.sleep);
      }
    return {drawItineraries: drawItineraries};
  };
}(jQuery));
