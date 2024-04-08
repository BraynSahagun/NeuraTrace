var modeloLetras = null;

// Tomar y configurar el canvas
var canvas = document.getElementById("bigcanvas");
var ctx1 = canvas.getContext("2d");
var smallcanvas = document.getElementById("smallcanvas");
var ctx2 = smallcanvas.getContext("2d");

function limpiar() {
  ctx1.clearRect(0, 0, canvas.width, canvas.height);
  drawingcanvas.clear(); // Asegúrate de que drawingcanvas esté definido correctamente
}

function predecirletra() {
  resample_single(canvas, 45, 45, smallcanvas);

  var imgData = ctx2.getImageData(0, 0, 45, 45);
  var arr = []; // El arreglo completo debe ser un arreglo plano
  for (var i = 0; i < imgData.data.length; i += 4) {
    arr.push(imgData.data[i] / 255); // R
    arr.push(imgData.data[i + 1] / 255); // G
    arr.push(imgData.data[i + 2] / 255); // B
    // No necesitamos el canal alpha (imgData.data[i + 3]), ya que no se usa en el modelo
  }

  // No es necesario anidar el arreglo dentro de otro arreglo, solo reformatearlo
  var tensor4 = tf.tensor4d(arr, [1, 45, 45, 3]); // Asegúrate de que esta es la forma esperada
  var resultados = modeloLetras.predict(tensor4).dataSync();
  var mayorIndice = resultados.indexOf(Math.max.apply(null, resultados));
  var letras = obtenerArregloLetras();
  var prediccionLetra = letras[mayorIndice];

  console.log("Predicción:", prediccionLetra);
  document.getElementById("resultado").innerHTML = prediccionLetra;
}

function resample_single(canvas, width, height, resize_canvas) {
  var width_source = canvas.width;
  var height_source = canvas.height;
  width = Math.round(width);
  height = Math.round(height);

  var ratio_w = width_source / width;
  var ratio_h = height_source / height;
  var ratio_w_half = Math.ceil(ratio_w / 2);
  var ratio_h_half = Math.ceil(ratio_h / 2);

  var ctx = canvas.getContext("2d");
  var ctx2 = resize_canvas.getContext("2d");
  var img = ctx.getImageData(0, 0, width_source, height_source);
  var img2 = ctx2.createImageData(width, height);
  var data = img.data;
  var data2 = img2.data;

  for (var j = 0; j < height; j++) {
    for (var i = 0; i < width; i++) {
      var x2 = (i + j * width) * 4;
      var weight = 0;
      var weights = 0;
      var weights_alpha = 0;
      var gx_r = 0;
      var gx_g = 0;
      var gx_b = 0;
      var gx_a = 0;
      var center_y = (j + 0.5) * ratio_h;
      var yy_start = Math.floor(j * ratio_h);
      var yy_stop = Math.ceil((j + 1) * ratio_h);
      for (var yy = yy_start; yy < yy_stop; yy++) {
        var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
        var center_x = (i + 0.5) * ratio_w;
        var w0 = dy * dy; //pre-calc part of w
        var xx_start = Math.floor(i * ratio_w);
        var xx_stop = Math.ceil((i + 1) * ratio_w);
        for (var xx = xx_start; xx < xx_stop; xx++) {
          var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
          var w = Math.sqrt(w0 + dx * dx);
          if (w >= 1) {
            //pixel too far
            continue;
          }
          //hermite filter
          weight = 2 * w * w * w - 3 * w * w + 1;
          var pos_x = 4 * (xx + yy * width_source);
          //alpha
          gx_a += weight * data[pos_x + 3];
          weights_alpha += weight;
          //colors
          if (data[pos_x + 3] < 255) weight = (weight * data[pos_x + 3]) / 250;
          gx_r += weight * data[pos_x];
          gx_g += weight * data[pos_x + 1];
          gx_b += weight * data[pos_x + 2];
          weights += weight;
        }
      }
      data2[x2] = gx_r / weights;
      data2[x2 + 1] = gx_g / weights;
      data2[x2 + 2] = gx_b / weights;
      data2[x2 + 3] = gx_a / weights_alpha;
    }
  }

  //Ya que esta, exagerarlo. Blancos blancos y negros negros..?

  for (var p = 0; p < data2.length; p += 4) {
    var gris = data2[p]; //Esta en blanco y negro

    if (gris < 100) {
      gris = 0; //exagerarlo
    } else {
      gris = 255; //al infinito
    }

    data2[p] = gris;
    data2[p + 1] = gris;
    data2[p + 2] = gris;
  }

  ctx2.putImageData(img2, 0, 0);
}

function obtenerArregloLetras() {
  const letras = [];
  // Agregar mayúsculas (A-Z)
  for (let i = 65; i <= 90; i++) {
    letras.push(String.fromCharCode(i));
  }
  // Agregar minúsculas (a-z)
  for (let i = 97; i <= 122; i++) {
    letras.push(String.fromCharCode(i));
  }
  return letras;
}

//Cargar modelo
(async () => {
  console.log("Cargando modelo...");
  modeloLetras = await tf.loadLayersModel("entrenamiento2.json");
  console.log("Modelo cargado...");
})();
