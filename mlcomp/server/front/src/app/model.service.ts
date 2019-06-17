import {Injectable} from "@angular/core";
import {BaseService} from "./base.service";

@Injectable({
    providedIn: 'root'
})
export class ModelService extends BaseService {
    protected collection_part: string = 'models';
    protected single_part: string = 'model';
}