document.getElementById("checkBtn").addEventListener("click", async () => {
  const text = document.getElementById("newsInput").value;
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  document.getElementById("result").innerText = `Prediction: ${data.prediction.toUpperCase()}`;
});
