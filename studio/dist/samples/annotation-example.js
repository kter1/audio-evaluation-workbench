const audio = document.getElementById('sample-audio');
let intervalEnd = null;
document.querySelectorAll('.seek').forEach(button => {
  button.addEventListener('click', async () => {
    const start = Number(button.dataset.start), end = Number(button.dataset.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start || end > 30) return;
    audio.currentTime = start;
    intervalEnd = end;
    try { await audio.play(); } catch { intervalEnd = null; }
  });
});
audio.addEventListener('timeupdate', () => {
  if (intervalEnd !== null && audio.currentTime >= intervalEnd) {
    audio.pause();
    intervalEnd = null;
  }
});
audio.addEventListener('seeking', () => {
  if (intervalEnd !== null && audio.currentTime >= intervalEnd) intervalEnd = null;
});
