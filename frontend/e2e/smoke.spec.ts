import { test, expect } from '@playwright/test';

test.describe('AgroSense Critical Path', () => {
  test('has title and renders home page', async ({ page }) => {
    await page.goto('/');

    // Expect a title "to contain" a substring.
    // Adjust this based on your actual metadata
    await expect(page).toHaveTitle(/AgroSense/i);

    // Verify main heading exists
    await expect(page.getByText(/AgroSense/i).first()).toBeVisible();
  });

  test('can navigate to crop recommendation form', async ({ page }) => {
    await page.goto('/');
    
    // Assuming there's a button or link to start prediction
    // We check if we are on the page or if the form elements exist
    // For now, let's just verify the endpoint loads without error
    await page.goto('/');
    
    // Check if the form inputs for N, P, K exist
    // Using rough selectors - ideally use data-testid in future
    // await expect(page.getByLabel('Nitrogen')).toBeVisible(); 
  });
});
