document.getElementById("learningRateSlider").oninput = function () {
  document.getElementById("learningRateText").value = this.value;
};

document.getElementById("learningRateText").oninput = function () {
  document.getElementById("learningRateSlider").value = this.value;
};

document.getElementById("iterationsSlider").oninput = function () {
  document.getElementById("iterationsText").value = this.value;
};

document.getElementById("iterationsText").oninput = function () {
  // Validate input to ensure it's a number and within the slider's range
  const iterations = parseInt(this.value, 10);
  if (!isNaN(iterations) && iterations >= 1 && iterations <= 1000) {
    document.getElementById("iterationsSlider").value = iterations;
  } else {
    // Optionally reset or alert if the text input is invalid
    alert("Iterations value must be a number between 1 and 1000.");
    this.value = document.getElementById("iterationsSlider").value; // Reset to slider value
  }
};

let scatterChart;
document.getElementById("runModel").onclick = function () {
  const learningRate = document.getElementById("learningRateSlider").value;
  const iterations = document.getElementById("iterationsSlider").value;
  document.getElementById("loading").style.display = "block";
  const data = {
    learningRate: parseFloat(learningRate),
    iterations: parseInt(iterations),
  };
  fetch("/run_model", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Success:", data);
      // Update the chart with the response data
      updateChart(data);
      document.getElementById("loading").style.display = "none";
    })
    .catch((error) => {
      console.error("Error:", error);
      document.getElementById("loading").style.display = "none";
    });
};

function updateChart(data) {
  //update accuracy
  document.getElementById("accuracy").textContent = data.accuracy + "%";
  //update the chart
  const ctx = document.getElementById("scatterChart").getContext("2d");
  if (scatterChart) {
    scatterChart.destroy(); // Destroy previous chart instance if exists
  }
  scatterChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Iris-Setosa",
          data: data.X_test.filter(
            (_, index) =>
              data.predictions[index] === 1 &&
              data.predictions[index] === data.y_test[index]
          ).map((point) => ({ x: point[0], y: point[1] })),
          backgroundColor: "blue",
        },
        {
          label: "Iris-Versicolour",
          data: data.X_test.filter(
            (_, index) =>
              data.predictions[index] === -1 &&
              data.predictions[index] === data.y_test[index]
          ).map((point) => ({ x: point[0], y: point[1] })),
          backgroundColor: "red",
        },
        {
          label: "Misclassified",
          data: data.X_test.filter(
            (_, index) => data.predictions[index] !== data.y_test[index]
          ).map((point) => ({ x: point[0], y: point[1] })),
          backgroundColor: "yellow",
        },
      ],
    },
    options: {
      scales: {
        x: {
          title: {
            display: true,
            text: "Sepal Length",
          },
        },
        y: {
          title: {
            display: true,
            text: "Sepal Width",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Perceptron Classification Results",
        },
        legend: {
          display: true,
          position: "top",
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const label = context.dataset.label || "";
              return `${label}: [${context.raw.x.toFixed(
                2
              )}, ${context.raw.y.toFixed(2)}]`;
            },
          },
        },
      },
    },
  });
}
