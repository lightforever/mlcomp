import {Component, OnInit} from '@angular/core';
import {Router, ActivatedRoute} from '@angular/router';
import {FormBuilder, FormGroup, Validators} from '@angular/forms';
import {HttpClient} from "@angular/common/http";
import {MessageService} from "../message.service";
import {AppSettings} from "../app-settings";

@Component({templateUrl: 'login.component.html'})
export class LoginComponent implements OnInit {
    loginForm: FormGroup;
    loading = false;
    submitted = false;
    returnUrl: string;
    invalid: boolean;

    constructor(
        private formBuilder: FormBuilder,
        private route: ActivatedRoute,
        private router: Router,
        protected http: HttpClient,
        protected messageService: MessageService
    ) {
        // redirect to home if already logged in
        if (localStorage.getItem('token')) {
            this.router.navigate(['/']);
        }
    }

    ngOnInit() {
        this.loginForm = this.formBuilder.group({
            token: ['', Validators.required]
        });

        // get return url from route parameters or default to '/'
        this.returnUrl = this.route.snapshot.queryParams['returnUrl'] || '/';
    }

    // convenience getter for easy access to form fields
    get f() {
        return this.loginForm.controls;
    }

    onSubmit() {
        this.submitted = true;

        // stop here if form is invalid
        if (this.loginForm.invalid) {
            return;
        }

        this.loading = true;
        let token = this.f.token.value;
        this.http.post(AppSettings.API_ENDPOINT + "token", {'token': token}).subscribe(
            data => {
                localStorage.setItem('token', token);
                this.invalid = false;
                this.router.navigate([this.returnUrl]);
            },
            error => {
                this.messageService.add(error);
                this.loading = false;
                this.invalid = true;
            });
    }
}
