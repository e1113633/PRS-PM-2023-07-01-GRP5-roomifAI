<script>
import roomObjects from './room_objects.json'
import Tags from "svelte-tags-input";

const ROOMS = {
	living: "Living Room",
	bedroom: "Bedroom",
	dining: "Dining Room",
}

let tags = [];
let room = 'living';
let prompt = '';
let imgSrc = null;
let loading = false;

function onAutocomplete(input = '') {
	const array = roomObjects[room || 'living'].filter((item) => {
		return item.toLowerCase().includes(input.toLowerCase())
	})
	return array;
}

function submit() {
	loading = true;
	const roomName = ROOMS[room || 'living'];
	const tagString = (tags || []).join(', ');

	fetch("./api/generate?" + new URLSearchParams({
		prompt: `${roomName} with ${tagString}`,
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
			<div class="block title">Select a room</div>
			<label class="block">
				<select name="prompt" bind:value={room} class="input" disabled={loading}>
					{#each Object.entries(ROOMS) as [key, label]}
						<option value={key}>{label}</option>
					{/each}
				</select>
			</label>

			<div class="input block">
				<Tags bind:tags={tags} addKeys={[9]}
					maxTags={100}
					splitWith={','}
					onlyUnique={true} 
					removeKeys={[27]}
					placeholder={"What is in this room?"}
					autoComplete={onAutocomplete}
					name={"tags"}
					id={"tags-id"}
					allowBlur={true}
					disable={loading}
					readonly={false}
					minChars={3}
					onlyAutocomplete
					labelText=" "
					labelShow
					onTagClick={tag => console.log(tag)}  />
			</div>

			<button type="submit" class={`submit_btn ${loading ? 'loading' : ''}`} disabled={loading}>
				{loading ? "Loading. This will take several minutes..." : "Generate!"}
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
		min-height: 100%;
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

	.title {
		color: white;
	}

	.container {
		padding: 2rem 2rem;
	}

	.input {
		width: 100%;
		max-width: 50rem;
		margin: 0.5rem auto;
	}

	.image {
		width: 360px;
		height: 360px;
		padding: 1rem;
	}

	.submit_btn {
		display: inline-block;
		padding: 1rem 2rem;
		width: 100%;
		max-width: 50rem;
		margin: 0.5rem 0;
	}

	.loading {
		transform: scale(1);
		animation: pulse 2s infinite;
	}

	@keyframes pulse {
	0% {
		transform: scale(0.95);
		box-shadow: 0 0 0 0 rgba(0, 0, 0, 0.7);
	}

	70% {
		transform: scale(1);
		box-shadow: 0 0 0 10px rgba(0, 0, 0, 0);
	}

	100% {
		transform: scale(0.95);
		box-shadow: 0 0 0 0 rgba(0, 0, 0, 0);
	}
}
</style>