<script>
const ROOMS = {
	living: "Living Room",
	bedroom: "Bedroom",
	dining: "Dining Room",
}

let room = 'living';
let prompt = '';
let imgSrc = null;
let loading = false;

function submit() {
	loading = true;
	fetch("./api/generate?" + new URLSearchParams({
		prompt: prompt,
		room: room
	}))
	.then(response => {
		return response.blob();
	})
	.then(blob => {
		imgSrc = URL.createObjectURL(blob);
		loading = false;
	})
}
</script>

<main>
	<div class="container">
		<form on:submit|preventDefault={submit}>
			<label class="block">
				<select name="prompt" bind:value={room} class="input" disabled={loading}>
					{#each Object.entries(ROOMS) as [key, label]}
						<option value={key}>{label}</option>
					{/each}
				</select>
			</label>
			

			<label class="block">
				<textarea name="prompt" bind:value={prompt} class="input" disabled={loading} />
			</label>

			<button type="submit" class="submit_btn" disabled={loading}>
				{loading ? "Loading..." : "Generate!"}
			</button>

			{#if imgSrc}
			<div><img src={imgSrc} alt={prompt} class="image"/></div>
			{/if}
		</form>
	</div>
</main>

<style>
	main {
		text-align: center;
		width: 100%;
		height: 100%;
		margin: 0 auto;
		background: rgb(2,0,36);
		background: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(3,29,87,1) 52%, rgba(0,0,0,1) 100%);
	}

	/* h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	} */

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}

	.container {
		padding: 2rem 2rem;
	}

	.input {
		width: 100%;
		max-width: 50rem;
		margin: 0.5rem 0;
	}

	.image {
		width: 512px;
		height: 512px;
	}

	.submit_btn {
		display: inline-block;
		padding: 1rem 2rem;
		width: 100%;
		max-width: 50rem;
		margin: 0.5rem 0;
	}
</style>