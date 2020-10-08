// Convert json to array of dictionaries
let resultArray = JSON.parse(jsonResult);
let actionTypes = {
    "empty": null,
    "accept": "Accept Offer",
    "no_contact": "No Contact",
    "reject": "Reject Offer"
};

// Allow the sorting by these date formats
$.fn.dataTable.moment( 'DD-MMM-YYYY' );
$.fn.dataTable.moment( 'MMM-YYYY' );

let ccTable = $('#example').DataTable({
    "data" : resultArray,
    "columns" : [
        { "data": "parent_choice_key", title: "Choice ID"},
        { "data" : "child_idno", "title" : "Child ID" },
        { "data" : "parent_contact", "title" : "Contact Details"},
        { "data" : "study_level", "title" : "Study Level" },
        { "data" : "enrolment_date", "title" : "Enrolment Date"},
        { "data" : "reg_date", "title" : "Registration Date"},
        { "data" : "likelihood", "title" : "Likelihood"},
        { "data": "cc_action", "title": "Action"}
    ],
    "paging": false,
    "columnDefs": [ {
          // Last column is select inputs
          "targets": -1,
          "render": function (data, type, row, meta){
                                  var $select = $(`<select id=${row["parent_choice_key"]} class="action"></select>`, {
                                  });
                                  $.each(actionTypes, function (k, v) {
                                      var $option = $("<option></option>", {
                                          "text": v,
                                          "value": k
                                      });
                                      if (data == v) {
                                          $option.attr("selected", "selected")
                                      }
                                      $select.append($option);
                                  });
                                  $('.action').change(function(){
                                       var value = $(this).val();
                                   });
                                  return $select.prop("outerHTML");
                                  }
       }]
});

// When submit data button is clicked, send data to ajax
document.getElementById('submit-data-button').onclick = function() {
    let tableData = document.getElementsByClassName("action");
    let formattedData = {};
    $.each(tableData, function(_, val){
        let selectText = $(`#${val.id} option:selected` ).text();
        if(selectText.length){
            formattedData[val.id] = $(`#${val.id} option:selected` ).text();
        }
    });
    let finalData = JSON.stringify(formattedData);

    $.ajax({
        url: "/update_data",
        type:'POST',
        headers: {
//            'X-CSRF-TOKEN': '{{ csrf_token() }}'
        },
        "dataType": "json",
        "data": finalData,
        "contentType": "application/json; charset=utf-8",
        success: function(result) {
            alert("Your changes have been committed to the database");
        },
        error: function(result) {
            alert('There is an error committing your changes. Please contact the IT team.');
        }
    })
};
