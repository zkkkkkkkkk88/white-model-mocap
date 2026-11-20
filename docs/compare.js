const upstream = 'https://raw.githubusercontent.com/toyxyz/toyxyz_RGBMocap/237e1f3b51d4d2add4f3654f49259e0afc70522e/docs/samples/';

function waitFor(video, event) {
  return new Promise((resolve, reject) => {
    const listeners = new AbortController();
    function finish(error) {
      clearTimeout(timeout);
      listeners.abort();
      if (error) reject(error);
      else resolve();
    }
    const timeout = setTimeout(() => finish(new Error('视频加载超时')), 12000);
    video.addEventListener(event, () => {
      finish();
    }, { signal: listeners.signal });
    video.addEventListener('error', () => {
      finish(new Error('无法加载视频'));
    }, { signal: listeners.signal });
  });
}

for (const player of document.querySelectorAll('section[data-offset]')) {
  const [reference, model] = player.querySelectorAll('video');
  const button = player.querySelector('button');
  const progress = player.querySelector('input');
  const timer = player.querySelector('output');
  const status = player.querySelector('.status');
  const offset = Number(player.dataset.offset);
  let referenceReady;

  function alignReference() {
    reference.currentTime = offset + model.currentTime;
  }

  function updateControls() {
    progress.value = model.currentTime;
    timer.value = `${model.currentTime.toFixed(1)} / 5.0 秒`;
  }

  button.addEventListener('click', async () => {
    if (!model.paused) {
      model.pause();
      return;
    }
    button.disabled = true;
    status.textContent = '正在加载参考视频…';
    try {
      if (!referenceReady) {
        referenceReady = waitFor(reference, 'loadedmetadata');
        reference.preload = 'auto';
        reference.src = upstream + reference.dataset.file;
      }
      await referenceReady;
      if (model.ended) model.currentTime = 0;
      alignReference();
      if (reference.seeking) await waitFor(reference, 'seeked');
      await Promise.all([reference.play(), model.play()]);
      button.textContent = '暂停';
      status.textContent = '';
    } catch {
      model.pause();
      reference.pause();
      reference.removeAttribute('src');
      reference.load();
      referenceReady = undefined;
      status.textContent = '播放失败。可重试，或打开下方的原片链接。';
    } finally {
      button.disabled = false;
    }
  });

  model.addEventListener('pause', () => {
    reference.pause();
    button.textContent = '播放';
  });
  model.addEventListener('timeupdate', () => {
    updateControls();
    if (!model.paused && !reference.seeking && Math.abs(reference.currentTime - offset - model.currentTime) > 0.08) {
      alignReference();
    }
  });
  progress.addEventListener('input', () => {
    model.currentTime = Number(progress.value);
    if (reference.readyState) alignReference();
    updateControls();
  });
}
