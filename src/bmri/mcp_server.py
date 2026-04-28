#!/usr/bin/env python3
import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import aiohttp
import httpx

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("bmri_mcp")

VIEWER_PORT = 8050
VIEWER_HOST = f"http://localhost:{VIEWER_PORT}"
PROJECT_ROOT = Path(__file__).parent.parent.parent


class BMRIViewer:
    """MCP Server for bMRI Viewer - manages viewer instance and tests all endpoints."""

    def __init__(self):
        self.process = None
        self.client = None
        self.test_data_dir = PROJECT_ROOT / "test" / "resources" / "Mrtc-Studie_Cartilage_transplantation_05_GAPF97478" / "20211125_0925" / "T1rho"

    async def __aenter__(self):
        await self.start_viewer()
        return self

    async def __aexit__(self, *args):
        await self.stop_viewer()

    async def start_viewer(self) -> dict:
        """Start the viewer server with test data."""
        logger.info(f"Starting bMRI viewer on port {VIEWER_PORT}...")

        # Kill any existing process
        await self._kill_existing()
        await asyncio.sleep(0.5)

        # Start new process
        self.process = subprocess.Popen(
            [
                "uv", "run",
                "-m", "bmri.viewer",
                str(self.test_data_dir),
                "--port", str(VIEWER_PORT),
                "--no-browser"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )

        # Wait for server to be ready
        max_retries = 30
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{VIEWER_HOST}/api/info", timeout=2)
                    if resp.status_code == 200:
                        logger.info("✓ Viewer started successfully")
                        self.client = httpx.AsyncClient()
                        return {"status": "ok", "port": VIEWER_PORT}
            except Exception as e:
                await asyncio.sleep(0.5)

        raise RuntimeError(f"Viewer failed to start after {max_retries} attempts")

    async def stop_viewer(self) -> dict:
        """Stop the viewer server."""
        logger.info("Stopping viewer...")
        if self.client:
            await self.client.aclose()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        return {"status": "stopped"}

    async def _kill_existing(self):
        """Kill any existing viewer process."""
        try:
            subprocess.run(
                f"pkill -f 'bmri.viewer' || true",
                shell=True,
                check=False,
                capture_output=True
            )
        except Exception as e:
            logger.debug(f"Could not kill existing process: {e}")

    async def api_call(self, endpoint: str, params: dict | None = None) -> dict:
        """Make an API call to the viewer."""
        try:
            url = f"{VIEWER_HOST}{endpoint}"
            resp = await self.client.get(url, params=params, timeout=10)
            resp.raise_for_status()

            if "image" in resp.headers.get("content-type", ""):
                return {
                    "status": "ok",
                    "endpoint": endpoint,
                    "content_type": resp.headers.get("content-type"),
                    "size_bytes": len(resp.content),
                }
            else:
                return {
                    "status": "ok",
                    "endpoint": endpoint,
                    "data": resp.json(),
                }
        except Exception as e:
            return {
                "status": "error",
                "endpoint": endpoint,
                "error": str(e),
            }

    async def test_all_endpoints(self) -> dict:
        """Test all viewer endpoints."""
        logger.info("Testing all endpoints...")
        results = {}

        # 1. Test /api/info
        info = await self.api_call("/api/info")
        results["info"] = info
        if info["status"] != "ok":
            logger.error(f"✗ /api/info failed: {info}")
            return results

        logger.info(f"✓ /api/info: {info['data']}")
        api_info = info["data"]

        # 2. Test /api/manifest
        manifest = await self.api_call("/api/manifest")
        results["manifest"] = manifest
        logger.info(f"✓ /api/manifest loaded")

        # 3. Test /api/stats
        stats = await self.api_call("/api/stats")
        results["stats"] = stats
        if stats["status"] == "ok":
            logger.info(f"✓ /api/stats: {len(stats['data'])} parameters")
        else:
            logger.error(f"✗ /api/stats failed: {stats}")

        # 4. Test overlay endpoint for each parameter
        slice_idx = 0
        results["overlays"] = {}
        for param in api_info.get("parameters", [])[:2]:  # Test first 2 params
            result = await self.api_call(
                f"/api/overlay/{slice_idx}",
                params={"param": param, "mask": True}
            )
            results["overlays"][param] = result
            if result["status"] == "ok":
                logger.info(f"✓ /api/overlay for {param}: {result['size_bytes']} bytes")
            else:
                logger.error(f"✗ /api/overlay for {param} failed: {result}")

        # 5. Test pixel endpoint
        cx, cy = api_info["shape"][0] // 2, api_info["shape"][1] // 2
        pixel = await self.api_call(f"/api/pixel/{cx}/{cy}/{slice_idx}")
        results["pixel"] = pixel
        if pixel["status"] == "ok":
            logger.info(f"✓ /api/pixel: got fit data")
        else:
            logger.error(f"✗ /api/pixel failed: {pixel}")

        # 6. Test mask_data endpoint
        if api_info.get("has_mask"):
            mask_data = await self.api_call(f"/api/mask_data/{slice_idx}")
            results["mask_data"] = mask_data
            if mask_data["status"] == "ok":
                logger.info(f"✓ /api/mask_data: {len(mask_data['data'].get('data', []))} pixels")
            else:
                logger.error(f"✗ /api/mask_data failed: {mask_data}")

        # 7. Test frontend loading
        frontend = await self.api_call("/")
        results["frontend"] = {
            "status": "ok" if frontend["status"] == "ok" else "error",
            "endpoint": "/",
            "size_bytes": frontend.get("size_bytes", 0),
        }
        logger.info(f"✓ Frontend HTML loaded: {frontend.get('size_bytes', 0)} bytes")

        return results

    async def test_frontend_integration(self) -> dict:
        """Test that frontend resources are accessible."""
        logger.info("Testing frontend integration...")
        results = {}

        resources = [
            "/static/index.html",
            "/static/viewer.js",
            "/static/style.css",
        ]

        for resource in resources:
            try:
                resp = await self.client.get(f"{VIEWER_HOST}{resource}", timeout=5)
                results[resource] = {
                    "status": "ok" if resp.status_code == 200 else "not_found",
                    "status_code": resp.status_code,
                    "size": len(resp.content),
                }
                if resp.status_code == 200:
                    logger.info(f"✓ {resource}: {len(resp.content)} bytes")
                else:
                    logger.error(f"✗ {resource}: HTTP {resp.status_code}")
            except Exception as e:
                results[resource] = {"status": "error", "error": str(e)}
                logger.error(f"✗ {resource}: {e}")

        return results


async def main():
    """Run comprehensive tests of the bMRI viewer."""
    logger.info("=" * 60)
    logger.info("bMRI Viewer MCP Server - Full Test Suite")
    logger.info("=" * 60)

    try:
        async with BMRIViewer() as viewer:
            # Test all endpoints
            endpoint_results = await viewer.test_all_endpoints()

            # Test frontend integration
            frontend_results = await viewer.test_frontend_integration()

            # Summary
            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)

            all_ok = all(r.get("status") == "ok" for r in endpoint_results.values())
            frontend_ok = all(r.get("status") == "ok" for r in frontend_results.values())

            if all_ok and frontend_ok:
                print("✓ ALL TESTS PASSED")
            else:
                print("✗ SOME TESTS FAILED")
                if not all_ok:
                    print("\nFailed endpoints:")
                    for name, result in endpoint_results.items():
                        if result.get("status") != "ok":
                            print(f"  - {name}: {result.get('error', 'unknown error')}")
                if not frontend_ok:
                    print("\nFailed frontend resources:")
                    for name, result in frontend_results.items():
                        if result.get("status") != "ok":
                            print(f"  - {name}: {result.get('error', 'unknown error')}")

            print("\nDetailed results saved to test_results.json")
            with open("test_results.json", "w") as f:
                json.dump({
                    "endpoints": endpoint_results,
                    "frontend": frontend_results,
                }, f, indent=2)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
