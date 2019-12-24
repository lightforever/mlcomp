import {Injectable} from "@angular/core";
import {BaseService} from "../base.service";
import {BaseResult, ModelAddData, ModelStartData} from "../models";
import {AppSettings} from "../app-settings";
import {catchError} from "rxjs/operators";

@Injectable({
    providedIn: 'root'
})
export class ModelService extends BaseService {
    protected collection_part: string = 'models';
    protected single_part: string = 'model';

    add(data: ModelAddData) {
        let message = `${this.constructor.name}.add`;
        let info = {
            'name': data.name,
            'task': data.task,
            'equations': data.equations,
            'project': data.project,
            'file': data.file,
            'fold': data.fold
        };
        let url = AppSettings.API_ENDPOINT + this.single_part + '/add';
        return this.http.post<BaseResult>(url, info).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    start_begin(data: ModelStartData) {
        let message = `${this.constructor.name}.start_begin`;
        let info = {
            'model_id': data.model_id
        };
        let url = AppSettings.API_ENDPOINT + this.single_part + '/start_begin';
        return this.http.post<ModelStartData>(url, info).pipe(
            catchError(this.handleError<ModelStartData>(
                message,
                new ModelStartData())
            )
        );
    }

    start_end(data: ModelStartData) {
        let message = `${this.constructor.name}.start_end`;
        let info = {
            'dag': data.dag.id,
            'pipe': data.pipe,
            'model_id': data.model_id
        };
        let url = AppSettings.API_ENDPOINT + this.single_part + '/start_end';
        return this.http.post<BaseResult>(url, info).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove(id: number) {
        let message = `${this.constructor.name}.remove`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/remove';
        return this.http.post<BaseResult>(url, {'id': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }
}