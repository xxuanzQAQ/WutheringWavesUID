import asyncio
import functools
import hashlib
import json
import random
import time
import uuid
from typing import Dict

import aiohttp
import cv2
import numpy as np
from cryptography.hazmat.primitives import padding, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class CaptchaError(Exception):
    pass


class CaptchaLoadError(CaptchaError):
    pass


class CaptchaVerifyError(CaptchaError):
    pass


PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDB45NNFhRGWzMFPn9I7k7IexS5
XviJR3E9Je7L/350x5d9AtwdlFH3ndXRwQwprLaptNb7fQoCebZxnhdyVl8Jr2J3
FZGSIa75GJnK4IwNaG10iyCjYDviMYymvCtZcGWSqSGdC/Bcn2UCOiHSMwgHJSrg
Bm1Zzu+l8nSOqAurgQIDAQAB
-----END PUBLIC KEY-----"""
_public_key = serialization.load_pem_public_key(PUBLIC_KEY_PEM.encode())
_AES_IV = b"0000000000000000"


def _aes_encrypt(content: str, aes_key: bytes) -> str:
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(_AES_IV))
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(content.encode("utf-8")) + padder.finalize()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return encrypted_data.hex()


def _get_pow(pow_detail: dict, captcha_id: str, lot_number: str) -> dict:
    temp = [
        pow_detail["version"],
        str(pow_detail["bits"]),
        pow_detail["hashfunc"],
        pow_detail["datetime"],
        captcha_id,
        lot_number,
    ]
    c = "|".join(temp) + "||"
    h = "".join(f"{int((1 + random.random()) * 65536):04x}"[1:] for _ in range(4))
    pow_string = c + h
    hasher = hashlib.md5 if pow_detail.get("hashfunc") == "md5" else hashlib.sha256
    return {"pow_msg": pow_string, "pow_sign": hasher(pow_string.encode()).hexdigest()}


def _get_w_param(load_data: dict, track: list, captcha_id: str) -> str:
    if not isinstance(_public_key, rsa.RSAPublicKey):
        raise TypeError("Public key is not a valid RSA key.")

    aes_key = random.randbytes(16)
    encrypted_key_hex = _public_key.encrypt(aes_key, PKCS1v15()).hex()
    pow_res = _get_pow(load_data["pow_detail"], captcha_id, load_data["lot_number"])

    left_pos = track[-1][0]
    passtime = track[-1][2]
    userresponse = left_pos / 1.0059466666666665 + 2

    payload = {
        "setLeft": left_pos,
        "passtime": passtime,
        "userresponse": userresponse,
        "device_id": "",
        "lot_number": load_data["lot_number"],
        "pow_msg": pow_res["pow_msg"],
        "pow_sign": pow_res["pow_sign"],
        "geetest": "captcha",
        "lang": "zh",
        "ep": "123",
        "biht": str(random.randint(1000000000, 2000000000)),
        "em": {"ph": 0, "cp": 0, "ek": "11", "wd": 1, "nt": 0, "si": 0, "sc": 0},
    }
    json_str = json.dumps(payload, separators=(",", ":"))
    encrypted_payload_hex = _aes_encrypt(json_str, aes_key)
    return encrypted_payload_hex + encrypted_key_hex


def _get_slide_distance(bg_img: np.ndarray, slice_img: np.ndarray) -> int:
    slice_gray = cv2.cvtColor(slice_img, cv2.COLOR_BGRA2GRAY)
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)

    slice_edge = cv2.Canny(slice_gray, 80, 200)
    bg_edge = cv2.Canny(bg_gray, 80, 200)

    result = cv2.matchTemplate(bg_edge, slice_edge, cv2.TM_CCOEFF_NORMED)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    offset_x = max_loc[0]

    # Visualization
    # h, w = slice_edge.shape
    # match_vis = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
    # cv2.rectangle(match_vis, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
    # print(f"  [图像识别] 纹理边缘匹配位置: {offset_x}px (置信度: {_max_val:.2f})")
    return offset_x


def _generate_track(distance: int) -> list:
    track = [[0, 0, 0]]
    current_x, t = 0, 0
    total_time = random.randint(500, 800)
    while current_x < distance:
        remaining = distance - current_x
        move_x = random.uniform(1, 3) if remaining < 20 else random.uniform(2, 9)
        current_x += move_x
        if current_x > distance:
            current_x = distance
        t += random.randint(10, 40)
        if t > total_time:
            t = total_time
        track.append([round(current_x), random.choice([-1, 0, 1]), t])
        if t > total_time and current_x < distance:
            track.append([distance, random.choice([-1, 0, 1]), t + 15])
            break
    if track[-1][0] != distance:
        track.append([distance, random.choice([-1, 0, 1]), track[-1][2] + 20])
    return track


class GeeTestSolver:
    CAPTCHA_ID = "3afb60f292fa803fa809114b9a89b3f5"

    def __init__(self):
        self._session = aiohttp.ClientSession()
        self._loop = asyncio.get_running_loop()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _download_image(self, url: str) -> np.ndarray:
        async with self._session.get(url) as resp:
            img_bytes = await resp.read()
        image_np = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)

    async def _send_load(self) -> Dict:
        url = "https://gcaptcha4.geetest.com/load"
        params = {
            "callback": f"geetest_{int(time.time() * 1000)}",
            "captcha_id": self.CAPTCHA_ID,
            "challenge": str(uuid.uuid4()),
            "client_type": "web",
            "risk_type": "slide",
            "lang": "zh",
        }
        try:
            async with self._session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                res_text = await resp.text()
            data = json.loads(res_text[22:-1])["data"]
            data["bg"] = "https://static.geetest.com/" + data["bg"]
            data["slice"] = "https://static.geetest.com/" + data["slice"]
            return data
        except (aiohttp.ClientError, json.JSONDecodeError, KeyError) as e:
            raise CaptchaLoadError(f"Failed to load captcha data: {e}") from e

    async def _get_validate(self, load_data: dict, w: str) -> Dict:
        url = "https://gcaptcha4.geetest.com/verify"
        params = {
            "callback": f"geetest_{int(time.time() * 1000)}",
            "lot_number": load_data["lot_number"],
            "captcha_id": self.CAPTCHA_ID,
            "process_token": load_data["process_token"],
            "client_type": "web",
            "risk_type": "slide",
            "payload_protocol": "1",
            "pt": "1",
            "payload": load_data["payload"],
            "w": w,
        }
        async with self._session.get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            res = await resp.text()
        return json.loads(res[22:-1])

    async def solve(self) -> Dict:
        load_data = await self._send_load()

        bg_img_task = asyncio.create_task(self._download_image(load_data["bg"]))
        slice_img_task = asyncio.create_task(self._download_image(load_data["slice"]))
        bg_img, slice_img = await asyncio.gather(bg_img_task, slice_img_task)

        distance = await self._loop.run_in_executor(
            None, functools.partial(_get_slide_distance, bg_img, slice_img)
        )
        track = _generate_track(distance)
        w_param_func = functools.partial(
            _get_w_param, load_data, track, self.CAPTCHA_ID
        )
        w_param = await self._loop.run_in_executor(None, w_param_func)

        verify_data = await self._get_validate(load_data, w_param)
        if (
            verify_data.get("status") == "success"
            and verify_data.get("data", {}).get("result") == "success"
        ):
            return verify_data["data"]["seccode"]
        else:
            raise CaptchaVerifyError(
                f"Captcha verification failed: {verify_data.get('data', verify_data)}"
            )
