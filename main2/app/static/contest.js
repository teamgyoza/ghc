/**
 * @fileoverview Misc js utility function for the judge.
 */

(function($) {

  $(document).ready(function() {
    $(document).find('form').each(function(index) {
      $(this).on('submit', function(evt) {
        $(this).find('input[type=file]').each(function(index) {
          if ($(this).val() == '') {
            evt.preventDefault();
            evt.stopPropagation();
            window.alert('You need to provide a file.');
          }
        });
      });
    });
  });
}(jQuery));
